# Prepareation of data for training and evaluation
#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────────
VIDEOS_DIR       = "dataset/raw/"
ANNOTATIONS_DIR  = "dataset/annotations/"
OUT_DIR          = "dataset/processed/"

IMAGES_OUT = os.path.join(OUT_DIR, "images")
MASKS_OUT  = os.path.join(OUT_DIR, "masks")
# ────────────────────────────────────────────────────────────────────────────────
def parse_cvat(xml_path):
    """
    Parse one CVAT XML, return { frame_fname: [ {label, points}, … ], … }.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ann = {}
    for img in root.findall("image"):
        name = img.attrib["name"]
        polys = []
        for poly in img.findall("polygon"):
            pts = [tuple(map(int, p.split(",")))
                   for p in poly.attrib["points"].split(";")]
            polys.append({"label": poly.attrib["label"], "points": pts})
        if polys:
            ann.setdefault(name, []).extend(polys)
    return ann

def build_colormap(labels):
    """
    Assigns a constant distinct color (RGB tuple) to each label.
    """
    # Define a list of fixed colors (extend as needed)
    fixed_colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),  (255, 255, 0),(255, 0, 255),(0, 255, 255),(128, 0, 128),(255, 255, 255),]
    cmap = {}
    for idx, lbl in enumerate(labels):
        cmap[lbl] = fixed_colors[idx % len(fixed_colors)]
    return cmap

def process_video(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    # 1) Load annotations for this video
    ann_file = os.path.join(ANNOTATIONS_DIR, f"{base}.xml")
    all_ann = {}
    if os.path.isfile(ann_file):
        parsed = parse_cvat(ann_file)
        for fname, polys in parsed.items():
            all_ann.setdefault(fname, []).extend(polys)
    else:
        print(f"[!] No annotation file for video '{base}' at {ann_file}. Masks will be blank.")

    # 2) Build color map
    all_labels = {poly["label"]
                  for polys in all_ann.values()
                  for poly in polys}
    colormap = build_colormap(all_labels)

    # 3) Extract frames & draw masks
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar  = tqdm(total=total, desc=f"Processing {base}")
    idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        pbar.update(1)

        fname = f"{base}_{idx:06d}.png"
        # 3a) save grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(IMAGES_OUT, fname), gray)

        # 3b) make blank color mask & fill polygons if any
        h, w = frame.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        for poly in all_ann.get(fname, []):
            pts   = np.array(poly["points"], np.int32)[None, ...]
            color = colormap[poly["label"]]
            cv2.fillPoly(mask, pts, color)
        cv2.imwrite(os.path.join(MASKS_OUT, fname), mask)

    pbar.close()
    cap.release()
    
def main():

    # ensure output folders exist
    os.makedirs(IMAGES_OUT, exist_ok=True)
    os.makedirs(MASKS_OUT, exist_ok=True)
    
    video_files = glob(os.path.join(VIDEOS_DIR, "*"))
    for vid in video_files:
        process_video(vid)

    print("All done!")
    print("Frames in:", IMAGES_OUT)
    print("Masks  in:", MASKS_OUT)

if __name__ == "__main__":
    main()