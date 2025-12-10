import torch
import logging
import cv2
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt

# Import utilities
from loader import get_model
from utils.video_utils import index_matrix_to_rgb, get_sorted_frames
from utils.image_utils import quantize_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_inference_on_video(checkpoint_path: str, raw_frames_dir: str, output_video_path: str,
                          fps: float = 30.0, use_cpu: bool = False, masked_frames_dir: str = None,
                          loss_graph_path: str = None):
    """
    Loads model, performs inference on all frames in a directory, and creates a video.

    Args:
        checkpoint_path: Path to the trained model weights.
        raw_frames_dir: Directory containing raw frame images (e.g., temp_frames/raw/recordings_XX_enrollmentYY_...).
        output_video_path: Path to save the output video (e.g., inference_output/subject_enrol.mp4).
        fps: Frames per second for the output video (default: 30.0).
        use_cpu: If True, force CPU inference even if CUDA is available.
        masked_frames_dir: Optional directory containing ground truth masked frames for loss calculation.
        loss_graph_path: Optional path to save the loss graph (e.g., inference_output/loss_graph.png).
    """
    # --- Setup ---
    device = torch.device("cpu" if use_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")

    # --- Setup Loss Tracking ---
    compute_loss = masked_frames_dir is not None
    if compute_loss:
        criterion = nn.CrossEntropyLoss()
        frame_losses = []
        masked_frames_dir = Path(masked_frames_dir)
        if not masked_frames_dir.exists():
            logging.warning(f"Masked frames directory not found: {masked_frames_dir}. Skipping loss computation.")
            compute_loss = False
        else:
            masked_frame_paths = get_sorted_frames(masked_frames_dir)
            logging.info(f"Found {len(masked_frame_paths)} masked frames for loss computation")

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
        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
            # Load frame
            raw_frame = cv2.imread(str(frame_path))
            if raw_frame is None:
                logging.warning(f"Could not load frame {frame_path}, skipping...")
                if compute_loss:
                    frame_losses.append(np.nan)
                continue

            # Convert to grayscale, normalize, add channel and batch dim (1, 1, H, W)
            gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            input_tensor = torch.from_numpy(gray_frame).float().unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)

            # Inference
            with torch.no_grad():
                # Model output is (N, C, H, W) -> (1, 8, H, W) logits
                logits = model(input_tensor)

                # Compute loss if ground truth is available
                if compute_loss and frame_idx < len(masked_frame_paths):
                    # Load ground truth mask as RGB (colored mask from COLOR_PALETTE)
                    gt_mask_path = masked_frame_paths[frame_idx]
                    gt_mask_img = cv2.imread(str(gt_mask_path), cv2.IMREAD_COLOR)

                    if gt_mask_img is not None:
                        # Convert RGB mask to class indices (H, W) using quantize_matrix
                        gt_mask_indices = quantize_matrix(gt_mask_img)

                        # Convert to tensor and move to device
                        gt_mask_tensor = torch.from_numpy(gt_mask_indices).long().unsqueeze(0).to(device)

                        # Compute loss
                        loss = criterion(logits, gt_mask_tensor)
                        frame_losses.append(loss.item())
                    else:
                        logging.warning(f"Could not load ground truth mask {gt_mask_path}")
                        frame_losses.append(np.nan)

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

    # --- Generate Loss Graph ---
    if compute_loss and len(frame_losses) > 0:
        # Filter out NaN values for statistics
        valid_losses = [loss for loss in frame_losses if not np.isnan(loss)]

        if len(valid_losses) > 0:
            avg_loss = np.mean(valid_losses)
            min_loss = np.min(valid_losses)
            max_loss = np.max(valid_losses)

            logging.info(f"\n--- LOSS STATISTICS ---")
            logging.info(f"Average Loss: {avg_loss:.4f}")
            logging.info(f"Min Loss: {min_loss:.4f}")
            logging.info(f"Max Loss: {max_loss:.4f}")
            logging.info(f"Valid frames: {len(valid_losses)}/{len(frame_losses)}")

            # Create loss graph
            plt.figure(figsize=(12, 6))
            frame_numbers = list(range(len(frame_losses)))
            plt.plot(frame_numbers, frame_losses, linewidth=1.5, alpha=0.8)
            plt.xlabel('Frame Number', fontsize=12)
            plt.ylabel('Loss (CrossEntropyLoss)', fontsize=12)
            plt.title(f'Per-Frame Loss Over Video\nAvg: {avg_loss:.4f}, Min: {min_loss:.4f}, Max: {max_loss:.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save graph
            if loss_graph_path is None:
                # Default: save next to the output video with _loss_graph suffix
                loss_graph_path = str(output_video_path).replace('.mp4', '_loss_graph.png')

            loss_graph_path = Path(loss_graph_path)
            loss_graph_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(loss_graph_path), dpi=150)
            plt.close()

            logging.info(f"Loss graph saved to: {loss_graph_path}")
        else:
            logging.warning("No valid loss values to plot")
    elif compute_loss:
        logging.warning("Loss computation was enabled but no losses were recorded")


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
    parser.add_argument('--masked_frames_dir', type=str, default=None,
                        help='Optional directory containing ground truth masked frames for loss calculation.')
    parser.add_argument('--loss_graph_path', type=str, default=None,
                        help='Optional path to save the loss graph PNG. If not specified, saves next to output video.')

    args = parser.parse_args()

    run_inference_on_video(args.checkpoint_path, args.raw_frames_dir,
                          args.output_video_path, args.fps, use_cpu=args.cpu,
                          masked_frames_dir=args.masked_frames_dir,
                          loss_graph_path=args.loss_graph_path)


