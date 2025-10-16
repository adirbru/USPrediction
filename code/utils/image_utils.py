import numpy as np


COLOR_PALETTE = np.array([
    (0, 0, 0),      # Black
    (0, 128, 0),    # Green
    (128, 0, 0),    # Red
    (0, 0, 128),    # Blue
    (128, 128, 0),  # Olive/Khaki
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (128, 128, 128) # Gray
])


def quantize_matrix(mask: np.ndarray, color_palette: np.ndarray = COLOR_PALETTE) -> np.ndarray:
    """
    Reads a mask image, quantizes its colors to the given palette, and 
    returns a Tensor of shape (H, W) where each pixel value is the 
    index (class label) of the closest color in the palette.
    """
    mask = np.asarray(mask)

    # If grayscale (H, W) or (H, W, 1), convert to 3-channel for distance calc.
    if mask.ndim == 2:
        mask = np.stack([mask, mask, mask], axis=-1)
    elif mask.ndim == 3 and mask.shape[2] == 1:
        mask = np.concatenate([mask, mask, mask], axis=2)

    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError("mask must be HxW or HxWx1 or HxWx3")

    # Normalize floats in [0,1] to 0-255
    if np.issubdtype(mask.dtype, np.floating):
        mask = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    H, W, _ = mask.shape
    pixels = mask.reshape(-1, 3).astype(np.int32)              # (N,3)
    palette = np.asarray(color_palette, dtype=np.int32)        # (P,3)

    # Compute squared distances (N, P) and argmin -> index for each pixel
    diff = pixels[:, None, :] - palette[None, :, :]            # (N, P, 3)
    d2 = np.sum(diff * diff, axis=2)                           # (N, P)
    idx = np.argmin(d2, axis=1)                                # (N,)

    # Lookup palette colors and reshape back to (H, W, 3)
    rgb = palette[idx].astype(np.uint8).reshape(H, W, 3)
    return rgb