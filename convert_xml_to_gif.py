import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import imageio.v2 as imageio
import matplotlib.cm as cm


ZIP_PATH = "XML_files_neuropathy (1).zip"
EXTRACT_DIR = "extracted_xml"
OUTPUT_DIR = "gifs"
GIF_DURATION = 0.08  # seconds per frame


def strip_namespace(elem):  #btshil asma2 l tags hata python yla9iya d8ri
    """
    Remove XML namespaces so tags like:
    {http://...}rollover
    become simply:
    rollover
    """
    for el in elem.iter():
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
#btjeeb 9imi ra9miyi mn xml w btraji3a integer , iza l 3onsor mish mawjood aw l 9imi ghalat bitraje3 default

def parse_cells_text(cells_text):
    """
    Convert the text inside <cells> into a numpy array of floats.
    """
    if not cells_text:
        return np.array([], dtype=float)

    values = []
    for token in cells_text.replace("\n", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            pass

    return np.array(values, dtype=float)
#bit7awel l nas mawjood dakhel <cells> la array ar9am l xml recorded pressure as parg and we want to converted to numbers to draw it

def build_patch_matrix(container):
    """
    Build a matrix from:
      - cell_count/x
      - cell_count/y
      - cells
    """
    cell_count = container.find("cell_count")
    width = parse_int(cell_count, "x", 0)
    height = parse_int(cell_count, "y", 0)

    cells_node = container.find("cells")
    values = parse_cells_text(cells_node.text if cells_node is not None else "")

    if width <= 0 or height <= 0 or values.size == 0:
        return None

    expected = width * height

    
    if values.size < expected:
        padded = np.zeros(expected, dtype=float)
        padded[:values.size] = values
        values = padded
    elif values.size > expected:
        values = values[:expected]

    return values.reshape((height, width))
#build matrix from pressure data it reads cell count/x and cell count/y and cells
#then it converts numbers to 2d shape
#why patch because sometimes only the part of the foot

def place_patch_on_full_grid(patch_container, full_h, full_w):
    """
    Reconstruct the full measurement grid using:
      - cell_begin/x
      - cell_begin/y
      - patch cell_count/x
      - patch cell_count/y
      - cells
    """
    patch = build_patch_matrix(patch_container)
    if patch is None:
        return None

    full_grid = np.zeros((full_h, full_w), dtype=float)

    cell_begin = patch_container.find("cell_begin")
    start_x = parse_int(cell_begin, "x", 0)
    start_y = parse_int(cell_begin, "y", 0)

    patch_h, patch_w = patch.shape

    end_y = min(start_y + patch_h, full_h)
    end_x = min(start_x + patch_w, full_w)

    usable_h = max(0, end_y - start_y)
    usable_w = max(0, end_x - start_x)

    if usable_h == 0 or usable_w == 0:
        return full_grid

    full_grid[start_y:end_y, start_x:end_x] = patch[:usable_h, :usable_w]
    return full_grid
#bthot l patch bmahalo l mazboot 3la full sensor grid l2n ayahan l data lal foot bt3mil save ka smallo part 
#fa hay l function bseer 3ndak soora kamli lal griid wel foot b mahala l mazboot

def get_full_grid_size(root):
    """
    Read full measurement grid size from the XML root.
    Default fallback = 40 x 64
    """
    cell_count = root.find("cell_count")
    full_w = parse_int(cell_count, "x", 40)
    full_h = parse_int(cell_count, "y", 64)
    return full_h, full_w
#it reads the full size of  the grid from l xml


def find_rollover_quants(root):
    """
    Find all frames stored as:
    rollover -> data -> quant
    """
    quants = []
    for rollover in root.findall(".//rollover"):
        data_node = rollover.find("data")
        if data_node is None:
            continue
        quants.extend(data_node.findall("quant"))
    return quants
#it return all the frames in the rollover

def normalize_frame(frame, vmax):
    if vmax <= 0:
        return np.zeros_like(frame)
    return np.clip(frame / vmax, 0.0, 1.0)

#bit7awel 9imet l da8et byn 0 / 1
#l2n colormap bada values normalized hata ylawen l sora
def frame_to_rgb(frame_norm):
    """
    Convert normalized frame to colored RGB image.
    """
    colored = cm.turbo(frame_norm)[:, :, :3]  # remove alpha
    rgb = (colored * 255).astype(np.uint8)
    return rgb
#bit7awel frame normalized to color photos RGP it use colormap names turbo 
#small values cold colors , big values hot colors



def xml_to_gif(xml_path, output_dir, duration=GIF_DURATION):
    """
    Convert one XML file into one GIF.
    """
#read the xml
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Important: remove namespace
    strip_namespace(root)
#tjeeb hajm l manasa
    full_h, full_w = get_full_grid_size(root)
#tjeeb kil l frames
    quants = find_rollover_quants(root)
#bt3mil skip iza ma la9it frame
    if not quants:
        print(f"[SKIP] No rollover frames found in: {xml_path}")
        return

    frames = []
    for quant in quants:
        grid = place_patch_on_full_grid(quant, full_h, full_w)
        if grid is not None:
            frames.append(grid)

    if not frames:
        print(f"[SKIP] No valid frames in: {xml_path}")
        return
#tihsob 23lq d8t b kil malaf
    vmax = max(float(np.max(f)) for f in frames)
    rgb_frames = []
#totba3 kil frame wtlawno
    for frame in frames:
        frame_norm = normalize_frame(frame, vmax)
        rgb = frame_to_rgb(frame_norm)

        # Optional rotation if needed:
        # rgb = np.rot90(rgb, k=1)
#tohfad kil sora dakhel list
        rgb_frames.append(rgb)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    gif_path = os.path.join(output_dir, f"{base_name}.gif")
#t3mil gif 
    imageio.mimsave(gif_path, rgb_frames, duration=duration)
    print(f"[OK] GIF created: {gif_path}")

#btfik d8t malaaf l zip
def extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[OK] Extracted ZIP to: {extract_dir}")

#bitdawer bl folder w bitjae3 kil l malafat
def find_all_xml_files(folder):
    xml_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return sorted(xml_files)


def main():
    if not os.path.exists(ZIP_PATH):
        print(f"[ERROR] ZIP file not found: {ZIP_PATH}")
        return

    extract_zip(ZIP_PATH, EXTRACT_DIR)

    xml_files = find_all_xml_files(EXTRACT_DIR)
    if not xml_files:
        print("[ERROR] No XML files found after extraction.")
        return

    print(f"[INFO] Found {len(xml_files)} XML files.")

    for xml_file in xml_files:
        try:
            xml_to_gif(xml_file, OUTPUT_DIR)
        except Exception as e:
            print(f"[ERROR] Failed on {xml_file}: {e}")

    print("\nDone. Check the 'gifs' folder.")


if __name__ == "__main__":
    main()