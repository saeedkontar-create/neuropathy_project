import os
import re
import csv
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


# =========================
# CONFIG
# =========================
INPUT_FOLDER = r"C:\Users\Lenovo\Desktop\neuropathy_project\extracted_xml\XML_files"
OUTPUT_DATASET_DIR = "dataset"
LABELS_CSV = "labels.csv"

UPSCALE_FACTOR = 12
SMOOTH_RESAMPLE = Image.BICUBIC


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
# EXTRACT STEPS
# each rollover = one step
# =========================
def extract_steps_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    strip_namespace(root)

    steps = []

    for rollover in root.findall(".//rollover"):
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

        if len(frames) > 0:
            steps.append(frames)

    return steps


# =========================
# COMMON BBOX FOR ONE STEP
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
# NORMALIZE + COLORIZE
# =========================
def normalize_frame(frame, vmax):
    if vmax <= 0:
        return np.zeros_like(frame)
    return np.clip(frame / vmax, 0.0, 1.0)


def frame_to_rgb(frame_norm):
    colored = cm.jet(frame_norm)[:, :, :3]
    rgb = (colored * 255).astype(np.uint8)
    return rgb


def upscale_rgb_image(img, factor=UPSCALE_FACTOR, resample=SMOOTH_RESAMPLE):
    pil_img = Image.fromarray(img)
    new_size = (img.shape[1] * factor, img.shape[0] * factor)
    pil_img = pil_img.resize(new_size, resample=resample)
    return pil_img


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

    # zero-pad patient id
    patient_id = str(patient_num).zfill(3)

    return patient_id, label, walk_id


# =========================
# SAVE ONE STEP AS PNG FRAMES
# =========================
def save_step_frames(step_frames, step_folder):
    os.makedirs(step_folder, exist_ok=True)

    cropped_frames = crop_frames_with_common_bbox(step_frames)
    if len(cropped_frames) == 0:
        return 0

    vmax = max(float(np.max(f)) for f in cropped_frames)

    count = 0
    for i, frame in enumerate(cropped_frames):
        norm = normalize_frame(frame, vmax)
        rgb = frame_to_rgb(norm)
        img = upscale_rgb_image(rgb)

        frame_path = os.path.join(step_folder, f"frame_{i:03d}.png")
        img.save(frame_path)
        count += 1

    return count


# =========================
# MAIN
# =========================
def process_dataset(input_folder, output_dataset_dir, labels_csv_path):
    os.makedirs(output_dataset_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".xml")]
    xml_files.sort()

    print(f"Found {len(xml_files)} XML files")

    # step numbering per patient across all walks
    patient_step_counter = {}

    rows = []

    for xml_file in xml_files:
        xml_path = os.path.join(input_folder, xml_file)
        patient_id, label, walk_id = parse_filename_info(xml_file)

        steps = extract_steps_from_xml(xml_path)

        if patient_id not in patient_step_counter:
            patient_step_counter[patient_id] = 0

        patient_folder = os.path.join(output_dataset_dir, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)

        print(f"Processing: {xml_file} -> {len(steps)} steps")

        for step_frames in steps:
            patient_step_counter[patient_id] += 1
            global_step_id = patient_step_counter[patient_id]

            step_folder = os.path.join(patient_folder, f"step_{global_step_id:03d}")
            num_frames = save_step_frames(step_frames, step_folder)

            if num_frames == 0:
                continue

            rows.append({
                "patient_id": patient_id,
                "step_id": f"{global_step_id:03d}",
                "label": label,
                "walk_id": walk_id,
                "source_xml": xml_file,
                "num_frames": num_frames,
            })

    # save CSV
    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "step_id", "label", "walk_id", "source_xml", "num_frames"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Dataset saved to: {output_dataset_dir}")
    print(f"Labels CSV saved to: {labels_csv_path}")


if __name__ == "__main__":
    process_dataset(INPUT_FOLDER, OUTPUT_DATASET_DIR, LABELS_CSV)