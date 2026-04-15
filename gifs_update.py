import os
import numpy as np
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


# =========================
# CONFIG
# =========================
INPUT_FOLDER = r"C:\Users\Lenovo\Desktop\neuropathy_project\extracted_xml\XML_files"
OUTPUT_FOLDER = "gifs_steps"

GIF_DURATION = 0.12
UPSCALE_FACTOR = 12          # bigger = smoother/larger GIF
SMOOTH_RESAMPLE = Image.BICUBIC   # or Image.BILINEAR


# =========================
# XML helpers
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
# Build matrix from one quant
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
# Extract steps: each rollover = one step
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
# Common bbox for one step
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
# Normalize and colorize
# =========================
def normalize_frame(frame, vmax):
    if vmax <= 0:
        return np.zeros_like(frame)
    return np.clip(frame / vmax, 0.0, 1.0)


def frame_to_rgb(frame_norm):
    colored = cm.jet(frame_norm)[:, :, :3]
    rgb = (colored * 255).astype(np.uint8)
    return rgb


# =========================
# Make all RGB images same size
# =========================
def pad_rgb_images_to_same_shape(images):
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    padded_images = []
    for img in images:
        h, w = img.shape[:2]
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        canvas[:h, :w, :] = img
        padded_images.append(canvas)

    return padded_images


# =========================
# Upscale smoothly for nicer GIF
# =========================
def upscale_rgb_image(img, factor=UPSCALE_FACTOR, resample=SMOOTH_RESAMPLE):
    pil_img = Image.fromarray(img)
    new_size = (img.shape[1] * factor, img.shape[0] * factor)
    pil_img = pil_img.resize(new_size, resample=resample)
    return np.array(pil_img)


# =========================
# Save one step as GIF
# =========================
def frames_to_gif(step_frames, output_path, duration=GIF_DURATION):
    if len(step_frames) == 0:
        return False

    cropped_frames = crop_frames_with_common_bbox(step_frames)
    if len(cropped_frames) == 0:
        return False

    vmax = max(float(np.max(f)) for f in cropped_frames)
    images = []

    for frame in cropped_frames:
        norm = normalize_frame(frame, vmax)
        rgb = frame_to_rgb(norm)
        rgb = upscale_rgb_image(rgb)   # smoother / less pixelated
        images.append(rgb)

    # final safety: same exact shape
    images = pad_rgb_images_to_same_shape(images)

    imageio.mimsave(output_path, images, duration=duration)
    return True


# =========================
# Process one XML
# =========================
def process_xml(xml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    steps = extract_steps_from_xml(xml_path)

    if len(steps) == 0:
        print(f"{base_name}: 0 steps found")
        return

    saved = 0
    failed = 0

    for i, step_frames in enumerate(steps, start=1):
        try:
            output_path = os.path.join(output_dir, f"{base_name}_step_{i}.gif")
            ok = frames_to_gif(step_frames, output_path)
            if ok:
                saved += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {base_name}_step_{i}: {e}")

    print(f"{base_name}: {saved} GIFs created, {failed} failed")


# =========================
# Process all XMLs
# =========================
def process_all_xml(folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(folder) if f.lower().endswith(".xml")]
    xml_files.sort()

    print(f"Found {len(xml_files)} XML files")

    for file in xml_files:
        xml_path = os.path.join(folder, file)
        print(f"Processing: {file}")
        process_xml(xml_path, output_dir)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    process_all_xml(INPUT_FOLDER, OUTPUT_FOLDER)