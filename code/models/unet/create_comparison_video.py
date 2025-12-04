#!/usr/bin/env python3
"""
Script to create a side-by-side comparison video showing:
1. Raw video
2. Raw masks
3. Model predictions

With a frame counter displayed above the video.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
import logging

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loader import get_model
from utils.video_utils import RawVideo, MaskedVideo, index_matrix_to_rgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    model = get_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from {checkpoint_path}")
    return model


def get_subject_videos(raw_video_dir: Path, masked_video_dir: Path, subject_id: str):
    """Get all video files for a given subject."""
    subject_videos = []
    
    # Find all videos for this subject
    for raw_video_path in raw_video_dir.glob(f"recordings_{subject_id}_*.mp4"):
        video_name = raw_video_path.stem
        masked_video_path = masked_video_dir / f"{video_name}_mask.mp4"
        
        if masked_video_path.exists():
            subject_videos.append((raw_video_path, masked_video_path))
            logging.info(f"Found video pair: {video_name}")
        else:
            logging.warning(f"Mask video not found for {video_name}")
    
    return subject_videos


def process_video_pair(raw_video_path: Path, masked_video_path: Path, 
                       model: torch.nn.Module, device: torch.device,
                       temp_dir: Path, output_video_path: Path, fps: int = 30):
    """
    Process a video pair to create side-by-side comparison video.
    
    Args:
        raw_video_path: Path to raw video
        masked_video_path: Path to masked video
        model: Trained model for inference
        device: Device to run inference on
        temp_dir: Temporary directory for frame extraction
        output_video_path: Path to save output video
        fps: Frames per second for output video
    """
    logging.info(f"Processing video: {raw_video_path.name}")
    
    # Extract frames from videos
    raw_video = RawVideo(path=raw_video_path)
    masked_video = MaskedVideo(path=masked_video_path)
    
    raw_frames_dir = temp_dir / "raw" / raw_video_path.stem
    masked_frames_dir = temp_dir / "masked" / raw_video_path.stem
    
    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    masked_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames if not already extracted
    if raw_frames_dir.exists() and list(raw_frames_dir.glob("frame_*.png")):
        logging.info("Loading existing raw video frames...")
        raw_video.frames = sorted(raw_frames_dir.glob("frame_*.png"), 
                                  key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    else:
        logging.info("Extracting raw video frames...")
        raw_video.split_to_frames(raw_frames_dir)
    
    if masked_frames_dir.exists() and list(masked_frames_dir.glob("frame_*.png")):
        logging.info("Loading existing masked video frames...")
        masked_video.frames = sorted(masked_frames_dir.glob("frame_*.png"),
                                     key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    else:
        logging.info("Extracting masked video frames...")
        masked_video.split_to_frames(masked_frames_dir)
    
    num_frames = min(len(raw_video.frames), len(masked_video.frames))
    logging.info(f"Processing {num_frames} frames...")
    
    # Get frame dimensions from first frame
    first_raw = cv2.imread(str(raw_video.frames[0]), cv2.IMREAD_GRAYSCALE)
    if first_raw is None:
        raise ValueError(f"Could not read first raw frame: {raw_video.frames[0]}")
    
    h, w = first_raw.shape
    logging.info(f"Frame dimensions: {w}x{h}")
    
    # Create video writer for output
    # Output will be 3 frames side by side, so width is 3*w
    # Plus space for frame counter at top
    counter_height = 50
    output_width = 3 * w
    output_height = h + counter_height
    
    # Use XVID codec for better Windows compatibility
    # XVID is well-supported on Windows and creates .avi files
    # Change extension to .avi for XVID codec
    output_video_path_avi = output_video_path.with_suffix('.avi')
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    video_writer = cv2.VideoWriter(str(output_video_path_avi), fourcc, fps, (output_width, output_height))
    
    if not video_writer.isOpened():
        # Fallback to mp4v if XVID doesn't work
        logging.warning("XVID codec not available, trying mp4v...")
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (output_width, output_height))
        if video_writer.isOpened():
            logging.warning("Using mp4v codec. Note: This may not play well on Windows. Consider converting with ffmpeg.")
        else:
            raise RuntimeError(f"Could not open video writer for {output_video_path}. Tried XVID and mp4v codecs.")
    else:
        # Update output path to use .avi extension
        output_video_path = output_video_path_avi
        logging.info(f"Using XVID codec, output will be: {output_video_path}")
    
    # Process each frame
    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            logging.info(f"Processing frame {frame_idx}/{num_frames}")
        
        # Load raw frame
        raw_frame = cv2.imread(str(raw_video.frames[frame_idx]), cv2.IMREAD_GRAYSCALE)
        if raw_frame is None:
            logging.warning(f"Could not read raw frame {frame_idx}, skipping...")
            continue
        
        # Convert to BGR for display
        raw_frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
        
        # Load mask frame
        mask_frame = cv2.imread(str(masked_video.frames[frame_idx]), cv2.IMREAD_COLOR)
        if mask_frame is None:
            logging.warning(f"Could not read mask frame {frame_idx}, skipping...")
            continue
        
        # Run inference to get prediction
        # Prepare input tensor: (1, 1, H, W) normalized
        input_tensor = torch.from_numpy(raw_frame).float().unsqueeze(0).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            mask_indices = logits.argmax(dim=1).squeeze(0)
        
        # Convert prediction to RGB
        pred_rgb = index_matrix_to_rgb(mask_indices)
        # Convert from RGB to BGR for OpenCV
        pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
        
        # Create side-by-side frame
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Place the three frames side by side
        combined_frame[counter_height:, 0:w] = raw_frame_bgr
        combined_frame[counter_height:, w:2*w] = mask_frame
        combined_frame[counter_height:, 2*w:3*w] = pred_bgr
        
        # Add frame counter at the top
        frame_text = f"Frame: {frame_idx:06d} / {num_frames-1:06d}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(frame_text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(combined_frame, (0, 0), (output_width, counter_height), bg_color, -1)
        
        # Draw text centered
        text_x = (output_width - text_width) // 2
        text_y = (counter_height + text_height) // 2
        cv2.putText(combined_frame, frame_text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Add labels below the counter
        label_y = counter_height - 10
        label_font_scale = 0.6
        label_thickness = 1
        cv2.putText(combined_frame, "Raw Video", (w//2 - 50, label_y), font, label_font_scale, text_color, label_thickness)
        cv2.putText(combined_frame, "Raw Masks", (w + w//2 - 50, label_y), font, label_font_scale, text_color, label_thickness)
        cv2.putText(combined_frame, "Predictions", (2*w + w//2 - 50, label_y), font, label_font_scale, text_color, label_thickness)
        
        # Write frame to video
        video_writer.write(combined_frame)
    
    video_writer.release()
    logging.info(f"Video saved to: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description='Create side-by-side comparison video')
    parser.add_argument('--exp_dir', type=str, 
                       default='experiments/4',
                       help='Path to experiment directory containing best_unet.pth')
    parser.add_argument('--subject_id', type=str, default='06',
                       help='Subject ID (e.g., 06)')
    parser.add_argument('--raw_video_dir', type=str, default='../../../dataset/raw',
                       help='Directory containing raw videos')
    parser.add_argument('--masked_video_dir', type=str, default='../../../dataset/masks',
                       help='Directory containing masked videos')
    parser.add_argument('--output_dir', type=str, default='./comparison_videos',
                       help='Directory to save output videos')
    parser.add_argument('--temp_dir', type=str, default='./temp_frames',
                       help='Temporary directory for frame extraction')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for output video')
    parser.add_argument('--video_name', type=str, default=None,
                       help='Specific video to process (e.g., recordings_06_enrollment01_multi_playing01). If not specified, processes all videos for the subject.')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU inference even if CUDA is available')
    
    args = parser.parse_args()
    
    # Setup paths
    exp_dir = Path(__file__).parent / args.exp_dir
    checkpoint_path = exp_dir / "best_unet.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    raw_video_dir = Path(__file__).parent / args.raw_video_dir
    masked_video_dir = Path(__file__).parent / args.masked_video_dir
    
    if not raw_video_dir.exists():
        raise FileNotFoundError(f"Raw video directory not found: {raw_video_dir}")
    if not masked_video_dir.exists():
        raise FileNotFoundError(f"Masked video directory not found: {masked_video_dir}")
    
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(__file__).parent / args.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cpu" if args.cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Get videos for subject
    if args.video_name:
        # Process specific video
        raw_video_path = raw_video_dir / f"{args.video_name}.mp4"
        masked_video_path = masked_video_dir / f"{args.video_name}_mask.mp4"
        
        if not raw_video_path.exists() or not masked_video_path.exists():
            raise FileNotFoundError(f"Video pair not found for {args.video_name}")
        
        video_pairs = [(raw_video_path, masked_video_path)]
    else:
        # Process all videos for subject
        video_pairs = get_subject_videos(raw_video_dir, masked_video_dir, args.subject_id)
    
    if not video_pairs:
        raise ValueError(f"No videos found for subject {args.subject_id}")
    
    # Process each video pair
    for raw_video_path, masked_video_path in video_pairs:
        output_video_path = output_dir / f"{raw_video_path.stem}_comparison.mp4"
        
        try:
            process_video_pair(
                raw_video_path=raw_video_path,
                masked_video_path=masked_video_path,
                model=model,
                device=device,
                temp_dir=temp_dir,
                output_video_path=output_video_path,
                fps=args.fps
            )
        except Exception as e:
            logging.error(f"Error processing {raw_video_path.name}: {e}", exc_info=True)
    
    logging.info("All videos processed successfully!")


if __name__ == "__main__":
    main()

