import numpy as np


COLOR_PALETTE = np.array([
    (0, 0, 0),      # Black (for background)
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 255, 255) # White
])


def quantize_matrix(mask: np.ndarray, color_palette: np.ndarray = COLOR_PALETTE) -> np.ndarray:
    """
    Reads a mask image, quantizes its colors to the given palette, and 
    returns a Tensor of shape (H, W) where each pixel value is the 
    index (class label) of the closest color in the palette.
    """
    # Ensure mask is processed as an (H*W, 3) array of pixels
    pixels = mask.reshape(-1, 3)
    
    # 2. Calculate squared Euclidean distance from every pixel to every palette color
    # The quantization logic is here, but we will return the index instead of the color itself.
    
    # Use int32 for difference calculation to avoid overflow when squaring large numbers
    diff = pixels[:, np.newaxis, :].astype(np.int32) - color_palette[np.newaxis, :, :].astype(np.int32)
    
    # Sum of squared differences for each pixel-color pair: (N, len(palette))
    squared_distances = np.sum(diff**2, axis=2)
    
    # 3. Find the index (class label) of the closest color for each pixel
    # This is the key change: we find the index directly.
    closest_color_indices = np.argmin(squared_distances, axis=1)
    
    # 4. Reshape the indices back to the original image dimensions (H, W)
    h, w, _ = mask.shape
    labels_matrix = closest_color_indices.reshape(h, w)
    
    # 5. Return as a long tensor of shape (H, W) for CrossEntropyLoss
    return labels_matrix