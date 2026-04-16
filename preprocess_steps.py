import os
import csv
import math
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from scipy.ndimage import rotate


# =========================
# CONFIG
# =========================
INPUT_FOLDER = r"C:\Users\Lenovo\Desktop\neuropathy_project\extracted_xml\XML_files"
OUTPUT_FOLDER = "preprocessed_dataset"
LABELS_CSV = "preprocessed_labels.csv"

FINAL_SIZE = (100, 100)   # width, height


# =========================
# XML HELPERS
# =========================
def strip_namespace(root):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def parse_int(node, key, default=0):
    if node is None:
        return default
    child = node.find(key)
    if child is None or child.text is None:
        return default
    try:
        return int(float(child.text.strip()))
    except ValueError:
        return default


def parse_float(node, key, default=0.0):
    if node is None:
        return default
    child = node.find(key)
    if child is None or child.text is None:
        return default
    try:
        return float(child.text.strip())
    except ValueError:
        return default


def parse_cells(text):
    values = []
    if text is None:
        return np.array(values, dtype=float)

    for token in text.replace("\n", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            pass

    return np.array(values, dtype=float)


# =========================
# BUILD MATRIX FROM QUANT
# =========================
def build_patch_matrix(container):
    cell_count = container.find("cell_count")
    w = parse_int(cell_count, "x", 0)
    h = parse_int(cell_count, "y", 0)

    cells_node = container.find("cells")
    values = parse_cells(cells_node.text if cells_node is not None else None)

    if w <= 0 or h <= 0 or values.size == 0:
        return None

    expected = w * h

    if values.size < expected:
        padded = np.zeros(expected, dtype=float)
        padded[:values.size] = values
        values = padded
    elif values.size > expected:
        values = values[:expected]

    return values.reshape((h, w))


# =========================
# EXTRACT STEP METADATA
# each rollover = one step
# =========================
def extract_steps_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    strip_namespace(root)

    steps = []

    # full grid size if available
    root_cell_count = root.find("cell_count")
    full_w = parse_int(root_cell_count, "x", 40)
    full_h = parse_int(root_cell_count, "y", 64)

    # try to get side from events if available
    event_sides = [e.findtext("side", default="unknown") for e in root.findall(".//event")]

    rollovers = root.findall(".//rollover")

    for idx, rollover in enumerate(rollovers):
        data_node = rollover.find("data")
        if data_node is None:
            continue

        frames = []
        for quant in data_node.findall("quant"):
            frame = build_patch_matrix(quant)
            if frame is None:
                continue
            if np.sum(frame) > 0:
                frames.append(frame)

        if len(frames) == 0:
            continue

        # best effort side assignment
        side = event_sides[idx] if idx < len(event_sides) else "unknown"

        # try to extract heel / toe coordinates from matching event
        heel_xy = None
        toe_xy = None
        if idx < len(root.findall(".//event")):
            ev = root.findall(".//event")[idx]
            heel = ev.find("heel")
            toe = ev.find("toe")
            if heel is not None:
                heel_xy = (parse_float(heel, "x", 0.0), parse_float(heel, "y", 0.0))
            if toe is not None:
                toe_xy = (parse_float(toe, "x", 0.0), parse_float(toe, "y", 0.0))

        steps.append({
            "frames": frames,
            "side": side.lower(),
            "heel_xy": heel_xy,
            "toe_xy": toe_xy,
            "full_grid_shape": (full_h, full_w),
        })

    return steps


# =========================
# FILENAME PARSING
# Example: GC_21_Marche2.xml
# =========================
def parse_filename_info(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    label = parts[0] if len(parts) > 0 else "UNKNOWN"
    patient_num = parts[1] if len(parts) > 1 else "000"
    walk_id = parts[2] if len(parts) > 2 else "UNKNOWN"

    patient_id = str(patient_num).zfill(3)
    return patient_id, label, walk_id


# =========================
# COMMON BBOX
# =========================
def get_common_bbox(frames):
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)

    union_mask = np.zeros((max_h, max_w), dtype=bool)

    for frame in frames:
        h, w = frame.shape
        mask = np.zeros((max_h, max_w), dtype=bool)
        mask[:h, :w] = frame > 0
        union_mask |= mask

    if not union_mask.any():
        return None

    rows = np.any(union_mask, axis=1)
    cols = np.any(union_mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return y_min, y_max, x_min, x_max


def crop_frames_with_common_bbox(frames):
    bbox = get_common_bbox(frames)
    if bbox is None:
        return []

    y_min, y_max, x_min, x_max = bbox
    cropped = []

    for frame in frames:
        h, w = frame.shape
        need_h = max(h, y_max + 1)
        need_w = max(w, x_max + 1)

        padded = np.zeros((need_h, need_w), dtype=frame.dtype)
        padded[:h, :w] = frame

        c = padded[y_min:y_max + 1, x_min:x_max + 1]
        cropped.append(c)

    return cropped


# =========================
# FPA
# =========================
def compute_fpa_from_heel_toe(heel_xy, toe_xy):
    """
    Returns angle in degrees.
    If toe/heel are unavailable, returns 0.
    """
    if heel_xy is None or toe_xy is None:
        return 0.0

    dx = toe_xy[0] - heel_xy[0]
    dy = toe_xy[1] - heel_xy[1]

    if dx == 0 and dy == 0:
        return 0.0

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


# =========================
# IMAGE OPS
# =========================
def rotate_frame(frame, angle_deg):
    """
    Rotate frame using scipy.ndimage.rotate.
    reshape=True keeps full content.
    """
    return rotate(frame, angle=angle_deg, reshape=True, order=1, mode="constant", cval=0.0)


def flip_left_to_right(frame):
    """
    Mirror horizontally.
    """
    return np.fliplr(frame)


def resize_bilinear(frame, size=FINAL_SIZE):
    """
    Resize using bilinear interpolation.
    """
    frame = np.array(frame, dtype=np.float32)

    # scale temporarily to 0-255 for PIL
    max_val = float(np.max(frame))
    if max_val > 0:
        img = (frame / max_val * 255.0).astype(np.uint8)
    else:
        img = np.zeros_like(frame, dtype=np.uint8)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, resample=Image.BILINEAR)

    arr = np.array(pil_img).astype(np.float32)

    # restore to approximate continuous scale
    if max_val > 0:
        arr = arr / 255.0 * max_val
    return arr


def zscore_normalize(frame):
    """
    Z-score normalization.
    """
    mean = np.mean(frame)
    std = np.std(frame)

    if std == 0:
        return np.zeros_like(frame, dtype=np.float32)

    return ((frame - mean) / std).astype(np.float32)


def to_uint8_for_save(frame):
    """
    Convert normalized float array to 0-255 grayscale for PNG saving.
    """
    min_val = float(np.min(frame))
    max_val = float(np.max(frame))

    if max_val - min_val == 0:
        out = np.zeros_like(frame, dtype=np.uint8)
    else:
        out = (frame - min_val) / (max_val - min_val)
        out = (out * 255.0).astype(np.uint8)

    return out


# =========================
# PREPROCESS ONE STEP
# =========================
def preprocess_step(step_dict, reference_angle):
    frames = step_dict["frames"]
    side = step_dict["side"]
    heel_xy = step_dict["heel_xy"]
    toe_xy = step_dict["toe_xy"]

    # 1) crop step consistently
    frames = crop_frames_with_common_bbox(frames)
    if len(frames) == 0:
        return []

    # 2) compute FPA
    step_angle = compute_fpa_from_heel_toe(heel_xy, toe_xy)

    # rotation needed to align with control reference
    rotation_needed = reference_angle - step_angle

    processed = []
    for frame in frames:
        # 3) align orientation using rotation
        frame = rotate_frame(frame, rotation_needed)

        # 4) left foot symmetry -> right
        if side == "left":
            frame = flip_left_to_right(frame)

        # 5) resize to 100x100 using bilinear interpolation
        frame = resize_bilinear(frame, FINAL_SIZE)

        # 6) z-score normalization
        frame = zscore_normalize(frame)

        processed.append(frame)

    return processed


# =========================
# COMPUTE REFERENCE ANGLE FROM GC
# =========================
def compute_reference_angle(input_folder):
    xml_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".xml")]
    xml_files.sort()

    gc_angles = []

    for xml_file in xml_files:
        patient_id, label, walk_id = parse_filename_info(xml_file)
        if label != "GC":
            continue

        xml_path = os.path.join(input_folder, xml_file)
        steps = extract_steps_from_xml(xml_path)

        for step in steps:
            angle = compute_fpa_from_heel_toe(step["heel_xy"], step["toe_xy"])
            gc_angles.append(angle)

    if len(gc_angles) == 0:
        print("[WARNING] No GC angles found. Using 0.0 as reference angle.")
        return 0.0

    ref_angle = float(np.mean(gc_angles))
    return ref_angle


# =========================
# SAVE STEP FRAMES
# =========================
def save_step_frames(step_frames, step_folder):
    os.makedirs(step_folder, exist_ok=True)

    count = 0
    for i, frame in enumerate(step_frames):
        out = to_uint8_for_save(frame)
        img = Image.fromarray(out)
        frame_path = os.path.join(step_folder, f"frame_{i:03d}.png")
        img.save(frame_path)
        count += 1

    return count


# =========================
# MAIN PIPELINE
# =========================
def process_dataset(input_folder, output_folder, labels_csv_path):
    os.makedirs(output_folder, exist_ok=True)

    # compute control-group reference angle first
    reference_angle = compute_reference_angle(input_folder)
    print(f"[INFO] Reference angle from GC: {reference_angle:.4f} degrees")

    xml_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".xml")]
    xml_files.sort()

    patient_step_counter = {}
    rows = []

    for xml_file in xml_files:
        xml_path = os.path.join(input_folder, xml_file)
        patient_id, label, walk_id = parse_filename_info(xml_file)

        steps = extract_steps_from_xml(xml_path)

        if patient_id not in patient_step_counter:
            patient_step_counter[patient_id] = 0

        patient_folder = os.path.join(output_folder, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)

        print(f"Processing: {xml_file} -> {len(steps)} steps")

        for step in steps:
            patient_step_counter[patient_id] += 1
            step_id = patient_step_counter[patient_id]

            processed_frames = preprocess_step(step, reference_angle)
            if len(processed_frames) == 0:
                continue

            step_folder = os.path.join(patient_folder, f"step_{step_id:03d}")
            num_frames = save_step_frames(processed_frames, step_folder)

            rows.append({
                "patient_id": patient_id,
                "step_id": f"{step_id:03d}",
                "label": label,
                "walk_id": walk_id,
                "side": step["side"],
                "source_xml": xml_file,
                "num_frames": num_frames,
                "reference_angle": round(reference_angle, 4),
                "step_angle": round(compute_fpa_from_heel_toe(step["heel_xy"], step["toe_xy"]), 4),
            })

    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "step_id",
                "label",
                "walk_id",
                "side",
                "source_xml",
                "num_frames",
                "reference_angle",
                "step_angle",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Preprocessed dataset saved to: {output_folder}")
    print(f"Labels CSV saved to: {labels_csv_path}")


if __name__ == "__main__":
    process_dataset(INPUT_FOLDER, OUTPUT_FOLDER, LABELS_CSV)