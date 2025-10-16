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
        return cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)


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
    """Mask-only augmentation that quantizes colors to the palette and returns colored mask (RGB/BGR)."""
    def __init__(self, color_palette: np.ndarray = COLOR_PALETTE):
        # This augmentation operates in-place on mask images and returns
        # a class-index matrix (H, W) so dataset code can directly convert
        # to a torch.LongTensor without re-quantizing.
        super().__init__(name='quantize_mask', in_place=True, supported_types=[MaskedVideo])
        self.color_palette = color_palette

    def apply(self, mask_img: np.ndarray) -> np.ndarray:
        """
        Quantize a colored mask image to class indices and return the index matrix

        Args:
            mask_img: Colored mask image (H, W, 3) or similar.

        Returns:
            np.ndarray: 2D array (H, W) of integer class indices (0..K-1).
        """
        # mask_img is expected to be a BGR color image; quantize_matrix returns indices
        indices = quantize_matrix(mask_img, self.color_palette)
        return indices


def get_augmentation(name: str, **kwargs) -> Augmentation:
    augmentations = {
        'resize': lambda: ResizeAugmentation(size=kwargs.get('resize_to', (240, 240))),
        'flip': FlipHorizontalAugmentation,
        'random_resize_crop': lambda: RandomResizeCropAugmentation(crop_percent=kwargs.get('random_resize_crop_percent', 20)),
        'quantize': QuantizeMaskAugmentation,
    }
    if name not in augmentations:
        raise ValueError(f"Unknown augmentation name: {name}")
    return augmentations[name]()
