import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Union, Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a palette of distinct BGR colors for mapping grayscale values
COLOR_PALETTE = [
    (0, 0, 0),      # Black (for background)
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
]

def get_sorted_frames(folder: Union[str, Path], extension: str = ".png") -> List[Path]:
    folder = Path(folder)
    frames = sorted(
        [f for f in folder.glob(f"*{extension}")],
        key=lambda x: int(''.join(filter(str.isdigit, x.stem)))
    )
    if not frames:
        logging.warning(f"No frames found with extension '{extension}' in {folder}")
    
    return frames

def read_images(raw_path: Path, seg_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw_img = cv2.imread(str(raw_path))
    seg_img = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    
    if raw_img is None:
        raise RuntimeError(f"Could not read raw frame {raw_path}")
    
    if seg_img is None:
        raise RuntimeError(f"Could not read segmentation {seg_path}")
    
    # Ensure same dimensions
    if raw_img.shape[:2] != seg_img.shape:
        seg_img = cv2.resize(seg_img, (raw_img.shape[1], raw_img.shape[0]))
    
    return raw_img, seg_img

def create_colored_mask(seg_img: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
    """
    Create a colored mask by mapping grayscale values to a color palette.

    Args:
        seg_img: Grayscale segmentation mask as a numpy array.
        color_map: Dictionary mapping grayscale values to color palette indices.

    Returns:
        Colored mask as a numpy array.
    """
    # Initialize colored mask with same shape as raw image (RGB)
    colored_mask = np.zeros((seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)
    
    # Map each unique grayscale value to a color
    unique_values = np.sort(np.unique(seg_img))
    for value in unique_values:
        if value == 0:  # Skip background
            continue
        if value not in color_map:
            color_map[value] = len(color_map) + 1  # Skip black (index 0)
        color = COLOR_PALETTE[color_map[value] % len(COLOR_PALETTE)]
        mask = seg_img == value
        colored_mask[mask] = color
    
    return colored_mask

def overlay_segmentations(raw_folder: Union[str, Path], seg_folder: Union[str, Path], 
                         output_folder: Union[str, Path], opacity: float = 0.5, 
                         extension: str = ".png") -> bool:
    """
    Overlay colored segmentation masks onto raw video frames and save to output folder.
    Each unique grayscale value in the mask is mapped to a distinct color.

    Args:
        raw_folder: Path to folder containing raw video frames.
        seg_folder: Path to folder containing segmentation masks.
        output_folder: Path to save composite images.
        opacity: Opacity of colored masks (0.0 = fully transparent, 1.0 = fully opaque).
        extension: File extension of frames (e.g., '.png').

    Returns:
        bool: True if processing successful, False otherwise.
    """
    try:
        raw_folder, seg_folder, output_folder = map(Path, (raw_folder, seg_folder, output_folder))
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Get sorted raw frames
        raw_frames = get_sorted_frames(raw_folder, extension)
        if not raw_frames:
            return False
        
        # Initialize color mapping for consistency across frames
        color_map = {}
        
        # Process each frame
        for raw_path in raw_frames:
            seg_path = seg_folder / raw_path.name
            if not seg_path.exists():
                logging.warning(f"Segmentation mask not found for {raw_path.name}")
                continue
            
            try:
                # Read and validate images
                raw_img, seg_img = read_images(raw_path, seg_path)
            except RuntimeError as e:
                logging.exception(str(e))
                continue
            
            colored_mask = create_colored_mask(seg_img, color_map)
            composite = cv2.addWeighted(raw_img, 1.0, colored_mask, opacity, 0.0)
            output_path = output_folder / raw_path.name
            cv2.imwrite(str(output_path), composite)
        
        logging.info(f"Composite images saved to {output_folder}")
        return True
    
    except Exception as e:
        logging.exception(f"Error processing frames: {str(e)}")
        return False

def create_video_from_frames(input_folder: Union[str, Path], fps: int = 10, 
                             extension: str = ".png") -> bool:
    """
    Create a video from indexed PNG frames in a folder.

    Args:
        input_folder: Path to folder containing PNG frames.
        fps: Frames per second for the output video.
        extension: File extension of frames (e.g., '.png').

    Returns:
        bool: True if video creation successful, False otherwise.
    """
    try:
        logging.info(f"Creating video from frames in {input_folder} with {fps} FPS")
        input_folder = Path(input_folder)
        output_video = input_folder / f"{input_folder.name}.mp4"
        
        # Get sorted frames
        frames = get_sorted_frames(input_folder, extension)
        if not frames:
            return False
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            logging.warning(f"Failed to read first frame: {frames[0]}")
            return False
        
        height, width, _ = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        # Write each frame to video
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logging.warning(f"Could not read frame {frame_path}")
                continue
            video_writer.write(frame)
        
        video_writer.release()
        logging.info(f"Video successfully created at {output_video}")
        return True
    
    except Exception as e:
        logging.exception(f"Error creating video")
        return False

if __name__ == "__main__":
    raw_folder = "/Users/adirbruchim/University/Technion/Year3/Semseter F/Projects/Vista/USPrediction/code/dataset/processed/frames/recordings_01_enrollment01_multi_playing01"
    seg_folder = "/Users/adirbruchim/University/Technion/Year3/Semseter F/Projects/Vista/USPrediction/xmem_output/recordings_01_enrollment01_multi_playing01/masks"
    output_folder = "/Users/adirbruchim/University/Technion/Year3/Semseter F/Projects/Vista/USPrediction/xmem_output/recordings_01_enrollment01_multi_playing01/overlays"
    
    overlay_segmentations(raw_folder, seg_folder, output_folder, opacity=0.15)
    create_video_from_frames(output_folder, fps=20)