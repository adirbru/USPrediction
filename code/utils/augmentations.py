import cv2
import numpy as np
import random
from typing import Tuple
from utils.image_utils import quantize_matrix, COLOR_PALETTE
from utils.video_utils import MaskedVideo, RawVideo

class Augmentation:
    """Base augmentation that operates on a single image."""
    def __init__(self, name: str = "augmentation", in_place: bool = False, supported_types: list = None):
        self.name = name
        self.in_place = in_place
        self.supported_types = supported_types if supported_types is not None else [MaskedVideo, RawVideo]
        self.suffix = f"_{self.name}"

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class PairAugmentation(Augmentation):
    """Augmentation that must be applied in a synchronized manner to (raw, mask).

    By default, it applies the same single-image transform to both images.
    """
    def apply_pair(self, raw: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.apply(raw), self.apply(mask)


class ResizeAugmentation(Augmentation):
    def __init__(self, size: Tuple[int, int] = (240, 240)):
        super().__init__(name='resize')
        self.size = size

    def apply(self, img: np.ndarray) -> np.ndarray:
        dtype = img.dtype
        img = img.astype(np.float32)
        resized = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        return resized.astype(dtype)


class FlipHorizontalAugmentation(PairAugmentation):
    def __init__(self):
        super().__init__(name='flipped')

    def apply(self, img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 1)


class RandomResizeCropAugmentation(PairAugmentation):
    def __init__(self, crop_percent: int = 20):
        """
        crop_percent: how much of the original width/height to potentially remove (0–100)
        """
        super().__init__(name='cropped')
        self.crop_percent = crop_percent

    def apply_pair(self, raw: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = raw.shape[:2]

        # compute new random crop size based on percent
        min_scale = 1.0 - (self.crop_percent / 100.0)
        scale = random.uniform(min_scale, 1.0)  # random crop scale between (1 - crop_percent%) and full size

        new_h = int(h * scale)
        new_w = int(w * scale)

        # random top-left corner for cropping
        y = random.randint(0, h - new_h)
        x = random.randint(0, w - new_w)

        raw_crop = raw[y:y + new_h, x:x + new_w]
        mask_crop = mask[y:y + new_h, x:x + new_w]

        # optionally resize back to original dimensions (to keep output size consistent)
        raw_resized = cv2.resize(raw_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        return raw_resized, mask_resized


class QuantizeMaskAugmentation(Augmentation):
    """Mask-only augmentation that quantizes colored masks to class indices.

    This is the ONLY place where quantization should happen during training.
    MaskedVideo.get_frame_matrix() returns colored BGR images (H, W, 3),
    and this augmentation converts them to class indices (H, W).
    """
    def __init__(self, color_palette: np.ndarray = COLOR_PALETTE):
        super().__init__(name='quantize_mask', in_place=True, supported_types=[MaskedVideo])
        self.color_palette = color_palette

    def apply(self, mask_img: np.ndarray) -> np.ndarray:
        """
        Quantize a colored mask image to class indices.

        Args:
            mask_img: Colored mask image (H, W, 3) in BGR format with uint8 values.

        Returns:
            np.ndarray: 2D array (H, W) of integer class indices (0..K-1).
        """
        # Ensure input is a colored image, not already quantized indices
        if mask_img.ndim != 3:
            raise ValueError(f"Expected 3D colored mask (H, W, 3), got shape {mask_img.shape}")

        # quantize_matrix converts colored BGR image to class indices
        indices = quantize_matrix(mask_img, self.color_palette)
        return indices


class GammaAugmentation(Augmentation):
    """Gamma correction augmentation for ultrasound images.
    
    Simulates different gain/brightness settings that occur in ultrasound imaging
    due to varying machine settings, operator preferences, or tissue characteristics.
    This helps the model become robust to intensity variations in real-world scenarios.
    
    Why it's good for ultrasound segmentation:
    - Ultrasound machines have adjustable gain/brightness controls that operators use
    - Different tissue types and depths require different gain settings
    - Helps model learn intensity-invariant features for better generalization
    - Prevents overfitting to specific brightness levels in training data
    """
    def __init__(self, gamma_range: Tuple[float, float] = (0.5, 2.0)):
        """
        Args:
            gamma_range: Tuple of (min_gamma, max_gamma) for random gamma selection.
                         Values < 1.0 darken the image (e.g., 0.5), > 1.0 brighten it (e.g., 2.0).
                         Note: The actual power applied is 1/gamma, so lower gamma = higher power = darker.
        """
        super().__init__(name='gamma', in_place=True, supported_types=[RawVideo])
        self.gamma_range = gamma_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random gamma correction to the image.
        
        Args:
            img: Input image (H, W) or (H, W, C) as uint8 [0, 255]
            
        Returns:
            Gamma-corrected image with same dtype and shape
        """
        # Randomly select gamma value from range
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        # Normalize to [0, 1] for gamma correction
        img_normalized = img.astype(np.float32) / 255.0
        
        # Apply gamma correction: I_out = I_in^(1/gamma)
        # Note: We use 1/gamma because we want gamma=2 to brighten (raise to 0.5 power)
        img_gamma = np.power(img_normalized, 1.0 / gamma)
        
        # Convert back to uint8
        img_gamma = np.clip(img_gamma * 255.0, 0, 255).astype(img.dtype)
        
        return img_gamma


class SpeckleNoiseAugmentation(Augmentation):
    """Speckle noise augmentation for ultrasound images.
    
    Simulates the inherent multiplicative speckle noise that is characteristic
    of ultrasound imaging. Speckle is caused by interference of scattered waves
    from sub-resolution scatterers and is a fundamental property of ultrasound images.
    
    Why it's good for ultrasound segmentation:
    - Speckle is an inherent property of all ultrasound images, not just noise
    - Different machines and frequencies produce different speckle patterns
    - Helps model learn to segment structures despite texture variations
    - Improves robustness to real-world ultrasound image quality variations
    - Prevents model from overfitting to smooth, noise-free training images
    """
    def __init__(self, noise_variance: float = 0.1):
        """
        Args:
            noise_variance: Variance of the multiplicative noise (0.0 to 1.0).
                           Higher values add more noise. Typical range: 0.05-0.2
        """
        super().__init__(name='speckle', in_place=True, supported_types=[RawVideo])
        self.noise_variance = noise_variance

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply multiplicative speckle noise to the image.
        
        Speckle noise follows a multiplicative model: I_noisy = I * (1 + n)
        where n is noise following a distribution typical of ultrasound speckle.
        
        Args:
            img: Input image (H, W) or (H, W, C) as uint8 [0, 255]
            
        Returns:
            Image with speckle noise added, same dtype and shape
        """
        # Convert to float for noise addition
        img_float = img.astype(np.float32)
        
        # Generate multiplicative noise
        # For ultrasound speckle, we use Gaussian noise scaled by variance
        # The noise is multiplicative: I_out = I_in * (1 + noise)
        noise = np.random.normal(0.0, self.noise_variance, size=img.shape)
        
        # Apply multiplicative noise
        img_noisy = img_float * (1.0 + noise)
        
        # Clip to valid range and convert back to original dtype
        img_noisy = np.clip(img_noisy, 0, 255).astype(img.dtype)
        
        return img_noisy


def get_augmentation(name: str, **kwargs) -> Augmentation:
    augmentations = {
        'resize': lambda: ResizeAugmentation(size=kwargs.get('resize_to', (240, 240))),
        'flip': FlipHorizontalAugmentation,
        'random_resize_crop': lambda: RandomResizeCropAugmentation(crop_percent=kwargs.get('random_resize_crop_percent', 20)),
        'quantize': QuantizeMaskAugmentation,
        'gamma': lambda: GammaAugmentation(gamma_range=kwargs.get('gamma_range', (0.5, 2.0))),
        'speckle': lambda: SpeckleNoiseAugmentation(noise_variance=kwargs.get('speckle_variance', 0.1)),
    }
    if name not in augmentations:
        raise ValueError(f"Unknown augmentation name: {name}")
    return augmentations[name]()
