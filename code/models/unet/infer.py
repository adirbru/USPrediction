import torch
import logging
import cv2
from pathlib import Path
import argparse

# Import utilities
from loader import get_model
from utils.video_utils import index_matrix_to_rgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_inference(checkpoint_path: str, raw_frame_path: str = None, output_dir: str = './inference_output'):
    """
    Loads model, performs inference on a single frame, and saves the resulting mask.
    
    Args:
        checkpoint_path: Path to the trained model weights.
        raw_frame_path: Optional path to a real raw frame image file (.png or .jpg).
        output_dir: Directory to save the predicted mask.
    """
    # --- Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # --- Load Model ---
    model = get_model()
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logging.info(f"Successfully loaded model weights from: {checkpoint_path}")
    except FileNotFoundError:
        logging.error(f"Checkpoint not found at {checkpoint_path}. Using uninitialized model. Please train first.")
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")

    model.to(device)
    model.eval()

    # --- Load Input Frame ---
    logging.info(f"Attempting to load real frame from {raw_frame_path}")
    # Need to simulate the logic from RawVideo.get_frame_matrix
    raw_frame = cv2.imread(raw_frame_path)
    # Convert to grayscale, normalize, add channel and batch dim (1, 1, H, W)
    gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    input_tensor = torch.from_numpy(gray_frame).float().unsqueeze(0).unsqueeze(0) / 255.0
    logging.info(f"Loaded real frame of shape {input_tensor.shape}")
    
    input_tensor = input_tensor.to(device)
    
    # --- Inference ---
    logging.info("Starting inference...")
    with torch.no_grad():
        # Model output is (N, C, H, W) -> (1, 8, H, W) logits
        logits = model(input_tensor)
        
        # Convert logits to class indices (argmax over the class dimension)
        # Resulting shape: (1, H, W)
        mask_indices = logits.argmax(dim=1)
        
        # Remove batch dimension: (H, W)
        final_mask_hw = mask_indices.squeeze(0)

    logging.info("Inference complete. Converting prediction to color mask.")

    # --- Post-processing and Saving ---
    # Convert class indices (0-7) to a color image matrix (H, W, 3)
    rgb_mask_matrix = index_matrix_to_rgb(final_mask_hw)
    
    # OpenCV requires BGR format for saving
    bgr_mask_matrix = rgb_mask_matrix[:, :, ::-1]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    output_filepath = output_path / "predicted_mask.png"
    cv2.imwrite(str(output_filepath), bgr_mask_matrix)
    
    logging.info(f"\n--- SUCCESS ---\nPredicted color mask saved to: {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run U-Net inference on a single frame.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the trained model checkpoint (e.g., ./best_unet.pth).')
    parser.add_argument('--raw_frame_path', type=str, default=None,
                        help='Path to a single raw input image file for testing (optional).')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='Directory to save the resulting predicted mask image.')
    
    args = parser.parse_args()
    
    run_inference(args.checkpoint_path, args.raw_frame_path, args.output_dir)
