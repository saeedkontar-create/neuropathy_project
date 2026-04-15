import os
import zipfile
import copy
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


ZIP_PATH = "XML_files_neuropathy (1).zip"
EXTRACT_DIR = "extracted_xml"

CLEANED_XML_DIR = "cleaned_xml_dataset"
PLOTS_DIR = "plots"
SMALL_SAMPLES_DIR = "small_patch_samples"

FRAME_SUMMARY_CSV = "frame_summary.csv"
XML_SUMMARY_CSV = "xml_summary.csv"
TEXT_REPORT = "analysis_report.txt"

SMALL_H_THRESHOLD = 7
SMALL_W_THRESHOLD = 7
SMALL_AREA_THRESHOLD = 49
MAX_SMALL_PATCH_IMAGES = 30


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
    if not text:
        return np.array([], dtype=float)

    values = []
    for token in text.replace("\n", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            pass
    return np.array(values, dtype=float)


def build_patch_matrix(container):
    """
    Build raw patch matrix from:
    - cell_count/x
    - cell_count/y
    - cells
    """
    cell_count = container.find("cell_count")
    w = parse_int(cell_count, "x", 0)
    h = parse_int(cell_count, "y", 0)

    cells_node = container.find("cells")
    if cells_node is None:
        return None, w, h, "missing_cells"

    values = parse_cells(cells_node.text)

    if w <= 0 or h <= 0:
        return None, w, h, "invalid_dimensions"

    if values.size == 0:
        return None, w, h, "empty_cells"

    expected = w * h

    if values.size < expected:
        padded = np.zeros(expected, dtype=float)
        padded[:values.size] = values
        values = padded
    elif values.size > expected:
        values = values[:expected]

    try:
        mat = values.reshape((h, w))
    except ValueError:
        return None, w, h, "reshape_error"

    return mat, w, h, None


def get_all_rollovers(root):
    return root.findall(".//rollover")


def get_all_quants_in_rollover(rollover):
    data_node = rollover.find("data")
    if data_node is None:
        return []
    return data_node.findall("quant")


def normalize_to_uint8(arr):
    if arr is None or arr.size == 0:
        return None
    max_val = float(np.max(arr))
    if max_val <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr / max_val) * 255.0
    return out.astype(np.uint8)


def save_patch_image(matrix, path, scale_up=20):
    img = normalize_to_uint8(matrix)
    if img is None:
        return
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((img.shape[1] * scale_up, img.shape[0] * scale_up), Image.NEAREST)
    pil_img.save(path)


def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def find_all_xml_files(folder):
    xml_files = []
    for root_dir, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".xml"):
                xml_files.append(os.path.join(root_dir, f))
    return sorted(xml_files)


def make_plots(frame_df, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    if frame_df.empty:
        return

    plot_df = frame_df.copy()
    plot_df["dim_label"] = plot_df["height"].astype(str) + "x" + plot_df["width"].astype(str)

    # 1) Top dimensions bar plot
    top_dims = plot_df["dim_label"].value_counts().head(20)

    plt.figure(figsize=(12, 6))
    top_dims.plot(kind="bar")
    plt.title("Top 20 Most Frequent Patch Dimensions")
    plt.xlabel("Dimension (HxW)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_dimensions_barplot.png"), dpi=200)
    plt.close()

    # 2) Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df["width"], plot_df["height"], alpha=0.5)
    plt.title("Patch Dimension Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "width_height_scatter.png"), dpi=200)
    plt.close()

    # 3) Area histogram
    plt.figure(figsize=(10, 6))
    plt.hist(plot_df["area"], bins=30)
    plt.title("Patch Area Distribution")
    plt.xlabel("Area (height x width)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "area_histogram.png"), dpi=200)
    plt.close()

    # 4) Threshold view
    plt.figure(figsize=(10, 6))
    plt.hist(plot_df["area"], bins=50)
    plt.axvline(SMALL_AREA_THRESHOLD, linestyle="--")
    plt.title("Area Distribution with Small-Patch Threshold")
    plt.xlabel("Area")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "area_threshold_view.png"), dpi=200)
    plt.close()


def process_one_xml(xml_path, frame_rows, xml_rows, counters):
    """
    Clean one XML:
    - remove empty / zero / invalid quants
    - save cleaned XML
    - record dimension info and summaries
    """
    xml_name = os.path.basename(xml_path)
    label = xml_name.split("_")[0] if "_" in xml_name else "UNKNOWN"

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        strip_namespace(root)

        rollovers = get_all_rollovers(root)

        if not rollovers:
            xml_rows.append({
                "xml_name": xml_name,
                "label": label,
                "status": "removed_no_rollover",
                "total_frames": 0,
                "kept_frames": 0,
            })
            counters["removed_xml"] += 1
            return

        total_frames_this_xml = 0
        kept_frames_this_xml = 0

        # Iterate through rollovers and clean quants in place
        for rollover in rollovers:
            data_node = rollover.find("data")
            if data_node is None:
                continue

            quants = list(data_node.findall("quant"))

            for idx, quant in enumerate(quants):
                total_frames_this_xml += 1
                counters["total_frames"] += 1

                patch, w, h, err = build_patch_matrix(quant)
                area = h * w if h > 0 and w > 0 else 0
                is_small = (
                    h <= SMALL_H_THRESHOLD
                    or w <= SMALL_W_THRESHOLD
                    or area <= SMALL_AREA_THRESHOLD
                )

                row = {
                    "xml_name": xml_name,
                    "label": label,
                    "frame_index": total_frames_this_xml - 1,
                    "height": h,
                    "width": w,
                    "area": area,
                    "is_small": is_small,
                    "status": None,
                }

                if err is not None or patch is None:
                    if err == "empty_cells":
                        counters["removed_empty"] += 1
                        row["status"] = "removed_empty"
                    else:
                        counters["removed_invalid"] += 1
                        row["status"] = f"removed_invalid_{err}"

                    frame_rows.append(row)
                    data_node.remove(quant)
                    continue

                if patch.size == 0:
                    counters["removed_empty"] += 1
                    row["status"] = "removed_empty"
                    frame_rows.append(row)
                    data_node.remove(quant)
                    continue

                if np.sum(patch) == 0:
                    counters["removed_zero"] += 1
                    row["status"] = "removed_zero"
                    frame_rows.append(row)
                    data_node.remove(quant)
                    continue

                if is_small and counters["saved_small_images"] < MAX_SMALL_PATCH_IMAGES:
                    img_name = f"{os.path.splitext(xml_name)[0]}_frame{total_frames_this_xml - 1}_patch_{h}x{w}.png"
                    img_path = os.path.join(SMALL_SAMPLES_DIR, img_name)
                    save_patch_image(patch, img_path, scale_up=25)
                    counters["saved_small_images"] += 1

                counters["kept_frames"] += 1
                kept_frames_this_xml += 1
                row["status"] = "kept"
                frame_rows.append(row)

        # After cleaning, check if still any quant remains
        remaining_quants = root.findall(".//rollover/data/quant")
        if len(remaining_quants) == 0:
            xml_rows.append({
                "xml_name": xml_name,
                "label": label,
                "status": "removed_all_frames_removed",
                "total_frames": total_frames_this_xml,
                "kept_frames": 0,
            })
            counters["removed_xml"] += 1
            return

        os.makedirs(CLEANED_XML_DIR, exist_ok=True)
        out_path = os.path.join(CLEANED_XML_DIR, xml_name)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

        xml_rows.append({
            "xml_name": xml_name,
            "label": label,
            "status": "saved",
            "total_frames": total_frames_this_xml,
            "kept_frames": kept_frames_this_xml,
        })
        counters["kept_xml"] += 1
        print(f"[OK] Saved cleaned XML: {xml_name}")

    except Exception as e:
        xml_rows.append({
            "xml_name": xml_name,
            "label": label,
            "status": f"error_{str(e)}",
            "total_frames": 0,
            "kept_frames": 0,
        })
        counters["removed_xml"] += 1
        print(f"[ERROR] {xml_name} -> {e}")


def main():
    if not os.path.exists(ZIP_PATH):
        print(f"[ERROR] ZIP not found: {ZIP_PATH}")
        return

    os.makedirs(CLEANED_XML_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(SMALL_SAMPLES_DIR, exist_ok=True)

    extract_zip(ZIP_PATH, EXTRACT_DIR)
    xml_files = find_all_xml_files(EXTRACT_DIR)

    if not xml_files:
        print("[ERROR] No XML files found.")
        return

    frame_rows = []
    xml_rows = []

    counters = {
        "total_xml": len(xml_files),
        "kept_xml": 0,
        "removed_xml": 0,
        "total_frames": 0,
        "kept_frames": 0,
        "removed_zero": 0,
        "removed_empty": 0,
        "removed_invalid": 0,
        "saved_small_images": 0,
    }

    for xml_path in xml_files:
        process_one_xml(xml_path, frame_rows, xml_rows, counters)

    frame_df = pd.DataFrame(frame_rows)
    xml_df = pd.DataFrame(xml_rows)

    frame_df.to_csv(FRAME_SUMMARY_CSV, index=False)
    xml_df.to_csv(XML_SUMMARY_CSV, index=False)

    kept_only_df = frame_df[frame_df["status"] == "kept"].copy()
    if not kept_only_df.empty:
        make_plots(kept_only_df, PLOTS_DIR)

    # Build report
    lines = []
    lines.append("XML DATASET CLEANING + ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total XML files: {counters['total_xml']}")
    lines.append(f"Saved cleaned XML files: {counters['kept_xml']}")
    lines.append(f"Removed XML files: {counters['removed_xml']}")
    lines.append("")
    lines.append(f"Total frames: {counters['total_frames']}")
    lines.append(f"Kept frames: {counters['kept_frames']}")
    lines.append(f"Removed zero frames: {counters['removed_zero']}")
    lines.append(f"Removed empty frames: {counters['removed_empty']}")
    lines.append(f"Removed invalid frames: {counters['removed_invalid']}")
    lines.append("")

    if not kept_only_df.empty:
        kept_only_df["dim_label"] = kept_only_df["height"].astype(str) + "x" + kept_only_df["width"].astype(str)

        lines.append("Top 20 dimensions:")
        for dim, count in kept_only_df["dim_label"].value_counts().head(20).items():
            lines.append(f"  {dim}: {count}")

        lines.append("")
        small_df = kept_only_df[kept_only_df["is_small"] == True]
        lines.append(f"Small kept patches count: {len(small_df)}")

        if len(small_df) > 0:
            lines.append("Top small dimensions:")
            for dim, count in (small_df["dim_label"].value_counts().head(20)).items():
                lines.append(f"  {dim}: {count}")

        lines.append("")
        lines.append("Interpretation:")
        lines.append(
            "Only zero, empty, or invalid matrices were removed from the XML dataset."
        )
        lines.append(
            "Very small matrices were not automatically removed because they may still correspond "
            "to meaningful localized plantar-pressure patches."
        )
        lines.append(
            "Plots and sample images were generated to inspect the distribution of dimensions and "
            "visually examine small patches before making any threshold-based exclusion decision."
        )

    with open(TEXT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nDone.")
    print(f"Cleaned XML dataset: {CLEANED_XML_DIR}")
    print(f"Plots folder: {PLOTS_DIR}")
    print(f"Small patch images: {SMALL_SAMPLES_DIR}")
    print(f"Frame summary CSV: {FRAME_SUMMARY_CSV}")
    print(f"XML summary CSV: {XML_SUMMARY_CSV}")
    print(f"Text report: {TEXT_REPORT}")


if __name__ == "__main__":
    main()