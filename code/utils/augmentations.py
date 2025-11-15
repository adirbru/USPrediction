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


class ResizeAugmentation(PairAugmentation):
    def __init__(self, size: Tuple[int, int] = (240, 240)):
        super().__init__(name='resize')
        self.size = size

    def apply_pair(self, raw: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raw_dtype = raw.dtype
        mask_dtype = mask.dtype

        raw_resized = cv2.resize(raw.astype(np.float32), self.size, interpolation=cv2.INTER_LINEAR)
        raw_resized = raw_resized.astype(raw_dtype)

        mask_input = mask
        if mask_input.ndim == 2:
            mask_input = mask_input.astype(np.float32)
        mask_resized = cv2.resize(mask_input, self.size, interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized.astype(mask_dtype)

        return raw_resized, mask_resized


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


class RandomGammaAugmentation(PairAugmentation):
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5)):
        super().__init__(name='random_gamma', in_place=True, supported_types=[RawVideo])
        self.gamma_range = gamma_range

    def _adjust_raw(self, img: np.ndarray) -> np.ndarray:
        original_dtype = img.dtype
        if np.issubdtype(original_dtype, np.floating):
            working = np.clip(img, 0.0, 1.0)
            img_uint8 = (working * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        gamma = random.uniform(*self.gamma_range)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
        adjusted = cv2.LUT(img_uint8, table)

        if np.issubdtype(original_dtype, np.floating):
            adjusted = adjusted.astype(np.float32) / 255.0
        return adjusted.astype(original_dtype, copy=False)

    def apply(self, img: np.ndarray) -> np.ndarray:
        return self._adjust_raw(img)

    def apply_pair(self, raw: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        adjusted_raw = self._adjust_raw(raw)
        return adjusted_raw, mask.copy()


class SpeckleNoiseAugmentation(PairAugmentation):
    def __init__(self, std: float = 0.05):
        super().__init__(name='speckle_noise', in_place=True, supported_types=[RawVideo])
        self.std = std

    def _apply_raw(self, img: np.ndarray) -> np.ndarray:
        dtype = img.dtype
        img_float = img.astype(np.float32)
        noise = np.random.randn(*img_float.shape).astype(np.float32) * self.std
        noisy = img_float + img_float * noise

        if np.issubdtype(dtype, np.integer):
            noisy = np.clip(noisy, 0, 255)
            return noisy.astype(dtype)

        noisy = np.clip(noisy, 0.0, 1.0)
        return noisy.astype(dtype)

    def apply(self, img: np.ndarray) -> np.ndarray:
        return self._apply_raw(img)

    def apply_pair(self, raw: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        noisy_raw = self._apply_raw(raw)
        return noisy_raw, mask.copy()


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


def get_augmentation(name: str, **kwargs) -> Augmentation:
    augmentations = {
        'resize': lambda: ResizeAugmentation(size=tuple(kwargs.get('resize_to') or (240, 240))),
        'flip': FlipHorizontalAugmentation,
        'random_resize_crop': lambda: RandomResizeCropAugmentation(crop_percent=kwargs.get('random_resize_crop_percent', 20)),
        'quantize': QuantizeMaskAugmentation,
        'random_gamma': lambda: RandomGammaAugmentation(gamma_range=tuple(kwargs.get('random_gamma_range') or (0.7, 1.5))),
        'speckle_noise': lambda: SpeckleNoiseAugmentation(std=kwargs.get('speckle_noise_std', 0.05) or 0.05),
    }
    if name not in augmentations:
        raise ValueError(f"Unknown augmentation name: {name}")
    return augmentations[name]()
