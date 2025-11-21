import torch
import logging
import cv2
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

# Import utilities
from loader import get_model
from utils.video_utils import index_matrix_to_rgb, get_sorted_frames

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_inference_on_video(checkpoint_path: str, raw_frames_dir: str, output_video_path: str, 
                          fps: float = 30.0, use_cpu: bool = False):
    """
    Loads model, performs inference on all frames in a directory, and creates a video.
    
    Args:
        checkpoint_path: Path to the trained model weights.
        raw_frames_dir: Directory containing raw frame images (e.g., temp_frames/raw/recordings_XX_enrollmentYY_...).
        output_video_path: Path to save the output video (e.g., inference_output/subject_enrol.mp4).
        fps: Frames per second for the output video (default: 30.0).
        use_cpu: If True, force CPU inference even if CUDA is available.
    """
    # --- Setup ---
    device = torch.device("cpu" if use_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")
    
    # --- Load Model ---
    model = get_model()
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Successfully loaded model weights from: {checkpoint_path}")
    except FileNotFoundError:
        logging.error(f"Checkpoint not found at {checkpoint_path}. Using uninitialized model. Please train first.")
        raise
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        raise

    model.to(device)
    model.eval()

    # --- Load Input Frames ---
    raw_frames_dir = Path(raw_frames_dir)
    if not raw_frames_dir.exists():
        raise ValueError(f"Raw frames directory not found: {raw_frames_dir}")
    
    frame_paths = get_sorted_frames(raw_frames_dir)
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {raw_frames_dir}")
    
    logging.info(f"Found {len(frame_paths)} frames to process")
    
    # --- Get frame dimensions from first frame ---
    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise ValueError(f"Could not load first frame: {frame_paths[0]}")
    
    height, width = first_frame.shape
    logging.info(f"Frame dimensions: {width}x{height}")
    
    # --- Setup Video Writer ---
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video_path}")
    
    # --- Process Frames ---
    logging.info("Starting inference on all frames...")
    processed_count = 0
    
    try:
        for frame_path in tqdm(frame_paths, desc="Processing frames"):
            # Load frame
            raw_frame = cv2.imread(str(frame_path))
            if raw_frame is None:
                logging.warning(f"Could not load frame {frame_path}, skipping...")
                continue
            
            # Convert to grayscale, normalize, add channel and batch dim (1, 1, H, W)
            gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            input_tensor = torch.from_numpy(gray_frame).float().unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)
            
            # Inference
            with torch.no_grad():
                # Model output is (N, C, H, W) -> (1, 8, H, W) logits
                logits = model(input_tensor)
                
                # Convert logits to class indices (argmax over the class dimension)
                # Resulting shape: (1, H, W)
                mask_indices = logits.argmax(dim=1)
                
                # Remove batch dimension: (H, W)
                final_mask_hw = mask_indices.squeeze(0).cpu()
            
            # Convert class indices to RGB color mask
            rgb_mask_matrix = index_matrix_to_rgb(final_mask_hw)
            
            # Write frame to video
            video_writer.write(rgb_mask_matrix)
            processed_count += 1
        
        logging.info(f"Processed {processed_count} frames successfully")
        
    finally:
        video_writer.release()
    
    logging.info(f"\n--- SUCCESS ---\nOutput video saved to: {output_video_path}")
    logging.info(f"Video: {width}x{height} @ {fps} fps, {processed_count} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run U-Net inference on all frames and create a video.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the trained model checkpoint (e.g., ./best_unet.pth).')
    parser.add_argument('--raw_frames_dir', type=str, required=True,
                        help='Directory containing raw input frames (e.g., temp_frames/raw/recordings_XX_enrollmentYY_...).')
    parser.add_argument('--output_video_path', type=str, required=True,
                        help='Path to save the output video (e.g., inference_output/subject_enrol.mp4).')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frames per second for the output video (default: 30.0).')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference even if CUDA is available.')
    
    args = parser.parse_args()
    
    run_inference_on_video(args.checkpoint_path, args.raw_frames_dir, 
                          args.output_video_path, args.fps, use_cpu=args.cpu)

