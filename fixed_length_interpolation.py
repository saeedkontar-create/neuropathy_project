import os
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


# =========================
# CONFIG
# =========================
INPUT_DATASET_DIR = "preprocessed_dataset"
INPUT_LABELS_CSV = "preprocessed_labels.csv"

OUTPUT_DATASET_DIR = "fixed_length_dataset"
OUTPUT_LABELS_CSV = "fixed_length_labels.csv"

IMAGE_EXT = ".png"
SPLINE_ORDER = 3   # cubic B-spline interpolation


# =========================
# LOAD ONE STEP AS 3D ARRAY
# shape = (T, H, W)
# =========================
def load_step_frames(step_folder):
    frame_files = sorted([
        f for f in os.listdir(step_folder)
        if f.lower().endswith(IMAGE_EXT)
    ])

    frames = []
    for f in frame_files:
        path = os.path.join(step_folder, f)
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        frames.append(arr)

    if len(frames) == 0:
        return None

    return np.stack(frames, axis=0)


# =========================
# SAVE STEP BACK TO PNGs
# =========================
def save_step_frames(volume, step_folder):
    os.makedirs(step_folder, exist_ok=True)

    for i in range(volume.shape[0]):
        frame = volume[i]

        min_val = float(frame.min())
        max_val = float(frame.max())

        if max_val - min_val == 0:
            out = np.zeros_like(frame, dtype=np.uint8)
        else:
            out = (frame - min_val) / (max_val - min_val)
            out = (out * 255).astype(np.uint8)

        img = Image.fromarray(out)
        img.save(os.path.join(step_folder, f"frame_{i:03d}.png"))


# =========================
# COMPUTE TARGET FRAMES = MEDIAN
# =========================
def compute_target_frames(labels_csv):
    nums = []

    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nums.append(int(row["num_frames"]))

    if len(nums) == 0:
        raise ValueError("No num_frames found in CSV.")

    target = int(np.median(nums))
    return target


# =========================
# 3D INTERPOLATION
# input  shape = (T, H, W)
# output shape = (target_T, H, W)
# =========================
def interpolate_3d_volume(volume, target_frames, spline_order=3):
    T, H, W = volume.shape

    if T == target_frames:
        return volume.copy()

    zoom_factors = (
        target_frames / T,  # time axis
        1.0,                # height
        1.0                 # width
    )

    out = zoom(volume, zoom=zoom_factors, order=spline_order)
    return out.astype(np.float32)


# =========================
# MAIN
# =========================
def process_dataset():
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    target_frames = compute_target_frames(INPUT_LABELS_CSV)
    print(f"[INFO] Target frames (median): {target_frames}")

    with open(INPUT_LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    updated_rows = []

    for row in rows:
        patient_id = row["patient_id"]
        step_id = row["step_id"]

        input_step_folder = os.path.join(
            INPUT_DATASET_DIR,
            f"patient_{patient_id}",
            f"step_{step_id}"
        )

        output_step_folder = os.path.join(
            OUTPUT_DATASET_DIR,
            f"patient_{patient_id}",
            f"step_{step_id}"
        )

        if not os.path.exists(input_step_folder):
            print(f"[WARNING] Missing folder: {input_step_folder}")
            continue

        volume = load_step_frames(input_step_folder)
        if volume is None:
            print(f"[WARNING] No frames in: {input_step_folder}")
            continue

        fixed_volume = interpolate_3d_volume(
            volume,
            target_frames=target_frames,
            spline_order=SPLINE_ORDER
        )

        save_step_frames(fixed_volume, output_step_folder)

        row["num_frames_before"] = volume.shape[0]
        row["num_frames_after"] = fixed_volume.shape[0]
        updated_rows.append(row)

        print(
            f"[OK] patient_{patient_id} step_{step_id}: "
            f"{volume.shape[0]} -> {fixed_volume.shape[0]}"
        )

    # save updated labels csv
    if len(updated_rows) > 0:
        fieldnames = list(updated_rows[0].keys())

        with open(OUTPUT_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

    print("\nDone.")
    print(f"Fixed-length dataset saved to: {OUTPUT_DATASET_DIR}")
    print(f"Updated CSV saved to: {OUTPUT_LABELS_CSV}")


if __name__ == "__main__":
    process_dataset()