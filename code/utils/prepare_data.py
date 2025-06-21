#!/usr/bin/env python3
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List
import logging

# ─── Configuration ──────────────────────────────────────────────────────────────
VIDEOS_DIR = "dataset/raw/"
ANNOTATIONS_DIR = "dataset/annotations/"
OUT_DIR = "dataset/processed/"

# Optional overlay settings
SAVE_OVERLAYS = True  # Set to True to save overlay masks on original frames
OVERLAY_ALPHA = 0.3   # Opacity of the mask overlay (0.0 = transparent, 1.0 = opaque)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnotationData:
    label: str
    rle: str
    bounding_box: List[int]  # [left, top, width, height]

def rle_to_mask(rle_string: str, width: int, height: int) -> np.ndarray:
    """
    Convert RLE (Run-Length Encoded) string to binary mask matrix.
    The RLE format alternates between run lengths of 0s and 1s.
    """
    rle_numbers = [int(x.strip()) for x in rle_string.split(',')]
    
    # Create flattened mask array
    mask_flat = np.zeros(width * height, dtype=np.uint8)
    
    # Decode RLE: alternating between run lengths of 0s and 1s
    current_pos = 0
    fill = 0  # Start with 0 (background)
    
    for run_length in rle_numbers:
        mask_flat[current_pos:current_pos + run_length] = fill
        current_pos += run_length
        fill ^= 1  # Toggle between 0 and 1
    
    # Reshape to 2D matrix (height x width)
    return mask_flat.reshape(height, width)

def parse_cvat_xml(xml_path: Path) -> Tuple[Dict[int, List[AnnotationData]], Tuple[int, int], List[str]]:
    """
    Parse CVAT XML file and return annotations by frame index.
    Returns: (frame_annotations, (height, width), labels)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get original image dimensions
    original_size_node = root.find('meta/original_size')
    width = int(original_size_node.find('width').text)
    height = int(original_size_node.find('height').text)
    
    # Get labels
    labels_node = root.find('meta/job/labels')
    labels = [label.find('name').text for label in labels_node.findall('label')]
    
    # Parse annotations by frame
    frame_annotations = defaultdict(list)
    
    for track in root.findall("track"):
        label = track.attrib["label"]
        for mask_tag in track.findall("mask"):
            frame_idx = int(mask_tag.attrib["frame"])
            
            rle = mask_tag.attrib['rle']
            m_left = int(mask_tag.attrib['left'])
            m_top = int(mask_tag.attrib['top'])
            m_width = int(mask_tag.attrib['width'])
            m_height = int(mask_tag.attrib['height'])
            
            annotation = AnnotationData(
                label=label,
                rle=rle,
                bounding_box=[m_left, m_top, m_width, m_height]
            )
            
            frame_annotations[frame_idx].append(annotation)
    
    return frame_annotations, (height, width), labels

def build_colormap(labels: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    Assigns a distinct color (RGB tuple) to each label.
    """
    fixed_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 128),
        (255, 165, 0),
        (255, 192, 203),
        (128, 128, 128)
    ]
    
    colormap = {}
    for idx, label in enumerate(labels):
        colormap[label] = fixed_colors[idx % len(fixed_colors)]
    
    return colormap

def create_mask_from_annotations(annotations: List[AnnotationData], 
                               image_shape: Tuple[int, int], 
                               colormap: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """
    Create a colored mask from annotations for a single frame.
    """
    height, width = image_shape
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for annotation in annotations:
        color = colormap[annotation.label]
        m_left, m_top, m_width, m_height = annotation.bounding_box
        
        # Decode RLE to binary mask
        binary_mask = rle_to_mask(annotation.rle, m_width, m_height)
        
        # Ensure mask fits within image bounds
        end_y = min(m_top + m_height, height)
        end_x = min(m_left + m_width, width)
        actual_height = end_y - m_top
        actual_width = end_x - m_left
        
        if actual_height > 0 and actual_width > 0:
            # Clip binary mask if necessary
            clipped_mask = binary_mask[:actual_height, :actual_width]
            
            # Apply color to mask
            roi = mask[m_top:end_y, m_left:end_x]
            roi[clipped_mask == 1] = color
    
    return mask

def create_overlay_mask(original_frame: np.ndarray, 
                       annotations: List[AnnotationData], 
                       image_shape: Tuple[int, int], 
                       colormap: Dict[str, Tuple[int, int, int]], 
                       alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay mask on the original frame with specified opacity.
    """
    # Create the colored mask
    mask = create_mask_from_annotations(annotations, image_shape, colormap)
    
    # Convert original frame to same shape as mask if needed
    if len(original_frame.shape) == 3:
        frame_bgr = original_frame
    else:
        frame_bgr = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
    
    # Create overlay by blending original frame with mask
    # Only blend where mask is non-zero
    mask_binary = np.any(mask > 0, axis=2)  # Create binary mask for blending
    
    overlay = frame_bgr.copy().astype(np.float32)
    mask_float = mask.astype(np.float32)
    
    # Apply blending only where mask exists
    overlay[mask_binary] = (1 - alpha) * frame_bgr[mask_binary] + alpha * mask_float[mask_binary]
    
    return overlay.astype(np.uint8)

def process_video(video_path: Path) -> None:
    """
    Process a single video: extract frames and create corresponding masks.
    """
    video_name = video_path.stem
    logging.info(f"Processing video: {video_name}")
    
    # Find corresponding annotation file
    annotation_path = Path(ANNOTATIONS_DIR) / f"{video_name}.xml"
    if not annotation_path.exists():
        raise FileNotFoundError(f"  Warning: No annotation file found for {video_name}")
    
    # Create output directories
    frames_dir = Path(OUT_DIR) / "frames" / video_name
    masks_dir = Path(OUT_DIR) / "masks" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Create overlay directory if needed
    if SAVE_OVERLAYS:
        overlays_dir = Path(OUT_DIR) / "overlays" / video_name
        overlays_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse annotations
    frame_annotations, image_shape, labels = parse_cvat_xml(annotation_path)
    colormap = build_colormap(labels)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"  Processing {total_frames} frames...")
    
    with tqdm(total=total_frames, desc=f"  {video_name}") as pbar:
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame as grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_filename = f"{frame_idx:06d}.png"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), gray_frame)
            
            # Create and save mask
            annotations = frame_annotations.get(frame_idx, [])
            mask = create_mask_from_annotations(annotations, image_shape, colormap)
            mask_path = masks_dir / frame_filename
            cv2.imwrite(str(mask_path), mask)
            
            # Create and save overlay if enabled
            if SAVE_OVERLAYS and annotations:  # Only create overlay if there are annotations
                overlay = create_overlay_mask(frame, annotations, image_shape, colormap, OVERLAY_ALPHA)
                overlay_path = overlays_dir / frame_filename
                cv2.imwrite(str(overlay_path), overlay)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    logging.info(f"  Completed: {frame_idx} frames processed")

def main():
    """
    Main processing pipeline.
    """
    logging.info("Starting video processing pipeline...")
    logging.debug(f"""
                  Videos directory: {VIDEOS_DIR}
                  Annotations directory: {ANNOTATIONS_DIR}
                  Output directory: {OUT_DIR}""")
    
    # Ensure output directory exists
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = glob(os.path.join(VIDEOS_DIR, '*.mp4'))
    logging.info(f"Found {len(video_files)} video files:")
    # Process each video
    for video_file in video_files:
        try:
            process_video(Path(video_file))
        except Exception:
            logging.exception(f"Couldn't process video {video_file}")
    
    logging.debug(f"""
                  Frames saved in: {OUT_DIR}/frames/
                  "Masks saved in: {OUT_DIR}/masks/""")
    if SAVE_OVERLAYS:
        logging.info(f"Overlays saved in: {OUT_DIR}/overlays/")

if __name__ == "__main__":
    main()