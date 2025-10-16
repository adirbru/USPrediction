import torch
import cv2
import os
from torch.utils.data import Dataset
from pathlib import Path
import logging
import re
import shutil
import numpy as np

from utils.video_utils import RawVideo, MaskedVideo, get_sorted_frames
from utils.augmentations import get_augmentation, PairAugmentation


logging.basicConfig(level=logging.INFO)


class DatasetVideo:
    def __init__(self, raw_video: Path, masked_video: Path, output_dir: Path):
        self.raw = RawVideo(path=raw_video)
        self.masked = MaskedVideo(path=masked_video)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = raw_video.stem
    def split_to_frames(self, preprocess: bool = False, in_place_aug_objects: list = None):
        """Extract frames from raw/masked videos. If in_place_aug_objects provided and
        frames are freshly created, apply augmentations before saving frames to disk.
        """
        raw_folder = self.output_dir / self.raw.name / self.name
        masked_folder = self.output_dir / self.masked.name / self.name

        # Determine single-frame augmentations applicable to raw/masked (exclude PairAugmentation)
        raw_augs = []
        masked_augs = []
        if in_place_aug_objects:
            for aug in in_place_aug_objects:
                if isinstance(aug, PairAugmentation):
                    continue
                if RawVideo in aug.supported_types:
                    raw_augs.append(aug)
                if MaskedVideo in aug.supported_types:
                    masked_augs.append(aug)

        created_folders = False
        # If frames already exist (preprocessed), just load them instead of extracting again
        if not preprocess and raw_folder.exists():
            # Use sorted lists to maintain frame order
            self.raw.frames = get_sorted_frames(raw_folder)
        else:
            raw_folder.mkdir(parents=True, exist_ok=True)
            self.raw.split_to_frames(raw_folder, single_frame_augs=raw_augs)
            created_folders = True

        if not preprocess and masked_folder.exists():
            self.masked.frames = get_sorted_frames(masked_folder)
        else:
            # Otherwise extract from videos
            try:
                masked_folder.mkdir(parents=True, exist_ok=True)
                self.masked.split_to_frames(masked_folder, single_frame_augs=masked_augs)
                if len(self.raw.frames) != len(self.masked.frames):
                    raise ValueError(f"Frame count mismatch in {self.name}, skipping...")

            except Exception as e:
                logging.error(f"Error processing masked video {self.masked.path}: {e}")
                if created_folders:
                    logging.error("removing created frames.")
                    # Cleaning up
                    shutil.rmtree(self.output_dir, ignore_errors=True)
                    self.raw.frames.clear()
                    self.masked.frames.clear()

    def __len__(self):
        return len(self.raw.frames)
    
    def __getitem__(self, index):
        return self.raw[index], self.masked[index]


class USSegmentationDataset(Dataset):
    def __init__(self, raw_video_dir: str, masked_video_dir: str,
                 temp_dir: str = "./temp_frames", preprocess: bool = False,
                 in_place_augmentations: list[str] = None,
                 enrichment_augmentations: list[str] = None,
                 augmentation_params: dict = None):
        """
        Dataset for ultrasound segmentation from video files.

        Args:
            raw_video_dir: Directory containing raw video files
            masked_video_dir: Directory containing masked video files
            temp_dir: Temporary directory to store extracted frames
            preprocess: If True, force re-extraction and preprocessing of frames
            in_place_augmentations: List of augmentation names to apply on-the-fly during __getitem__
            enrichment_augmentations: List of augmentation names to apply during preprocessing to enrich dataset
            augmentation_params: Dictionary of parameters for augmentations
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.videos: list[DatasetVideo] = []
        self.subject_to_frames: dict[str, range] = {}
        self.preprocess = preprocess
        self.in_place_augmentations = in_place_augmentations if in_place_augmentations else []
        self.enrichment_augmentations = enrichment_augmentations if enrichment_augmentations else []
        self.augmentation_params = augmentation_params if augmentation_params else {}

        # Initialize augmentation objects for in-place augmentations
        self.in_place_aug_objects = []
        for aug_name in self.in_place_augmentations:
            try:
                aug = get_augmentation(aug_name, **self.augmentation_params)
                self.in_place_aug_objects.append(aug)
                logging.info(f"Loaded in-place augmentation: {aug_name}")
            except Exception as e:
                logging.error(f"Failed to load augmentation '{aug_name}': {e}")

        # Process videos to extract frames and apply enrichment augmentations
        self._process_videos(raw_video_dir, masked_video_dir)

        logging.info(f"Dataset initialized with {len(self.videos)} video pairs.")

    def _process_videos(self, raw_video_dir: str, masked_video_dir: str):
        """Process video files to extract frames and organize by subject"""
        logging.info("Processing videos to extract frames...")

        for video_name in os.listdir(raw_video_dir):
            try:
                raw_video_path = Path(raw_video_dir) / video_name
                masked_video_path = Path(masked_video_dir) / video_name

                if not raw_video_path.exists() or not masked_video_path.exists():
                    logging.warning(f"Video pair not found for {video_name}, skipping...")
                    continue

                dataset_video = DatasetVideo(raw_video=raw_video_path,
                                            masked_video=masked_video_path,
                                            output_dir=self.temp_dir,
                                            )
                dataset_video.split_to_frames(preprocess=self.preprocess, in_place_aug_objects=self.in_place_aug_objects)

                # Extract subject ID from video name (e.g., recordings_01_enrollment01... -> subject 01)
                subject_id = self._extract_subject_id(video_name)
                if subject_id is None:
                    logging.warning(f"Could not extract subject ID from {video_name}, skipping...")
                    continue

                # Add the original video frames
                last_frame = max([rng.stop for rng in self.subject_to_frames.values()], default=0)
                self.subject_to_frames[subject_id] = range(last_frame, last_frame + len(dataset_video))
                self.videos.append(dataset_video)

                # Handle enrichment augmentations
                # Always create augmentations if they don't exist, or load if they do
                if self.enrichment_augmentations:
                    self._apply_enrichment_augmentations(dataset_video, subject_id, raw_video_path)

            except Exception:
                logging.exception(f"Error processing video {video_name}, skipping...")
    
    def _apply_enrichment_augmentations(self, dataset_video: DatasetVideo, subject_id: str, raw_video_path: Path):

        """Apply enrichment augmentations to create additional augmented versions of the dataset"""
        for aug_name in self.enrichment_augmentations:
            try:
                aug = get_augmentation(aug_name, **self.augmentation_params)

                # Create augmented output directories
                aug_raw_folder = self.temp_dir / RawVideo.name / f"{raw_video_path.stem}{aug.suffix}"
                aug_masked_folder = self.temp_dir / MaskedVideo.name / f"{raw_video_path.stem}{aug.suffix}"

                # Check if augmented frames already exist
                augmented_frames_exist = (aug_raw_folder.exists() and
                                         aug_masked_folder.exists() and
                                         len(list(aug_raw_folder.glob("*.png"))) > 0)

                # If preprocess is enabled, force re-creation even if they exist
                should_create = not augmented_frames_exist or self.preprocess

                if should_create:
                    # Create (or re-create) augmented frames
                    if self.preprocess and augmented_frames_exist:
                        logging.info(f"Re-creating augmented frames for {aug_name} on {dataset_video.name} (preprocess mode)...")
                    else:
                        logging.info(f"Creating augmented frames for {aug_name} on {dataset_video.name}...")

                    # Create directories
                    aug_raw_folder.mkdir(parents=True, exist_ok=True)
                    aug_masked_folder.mkdir(parents=True, exist_ok=True)

                    # Apply augmentation to each frame pair
                    augmented_raw_frames = []
                    augmented_masked_frames = []

                    for frame_idx in range(len(dataset_video)):
                        raw_frame_path, masked_frame_path = dataset_video[frame_idx]

                        # Load frames as numpy arrays
                        raw_img = cv2.imread(str(raw_frame_path))
                        masked_img = cv2.imread(str(masked_frame_path))

                        # Apply augmentation
                        if isinstance(aug, PairAugmentation):
                            aug_raw, aug_masked = aug.apply_pair(raw_img, masked_img)
                        else:
                            # For non-pair augmentations, apply to both separately
                            aug_raw = aug.apply(raw_img)
                            aug_masked = aug.apply(masked_img)

                        # Save augmented frames
                        aug_raw_path = aug_raw_folder / f"frame_{frame_idx:06d}.png"
                        aug_masked_path = aug_masked_folder / f"frame_{frame_idx:06d}.png"

                        cv2.imwrite(str(aug_raw_path), aug_raw)
                        cv2.imwrite(str(aug_masked_path), aug_masked)

                        augmented_raw_frames.append(aug_raw_path)
                        augmented_masked_frames.append(aug_masked_path)

                    logging.info(f"Created {len(augmented_raw_frames)} augmented frames with {aug_name}")
                else:
                    # Augmented frames exist and we're not in preprocess mode, just load them
                    logging.info(f"Loading existing augmented frames for {aug_name} on {dataset_video.name}")
                    augmented_raw_frames = get_sorted_frames(aug_raw_folder)
                    augmented_masked_frames = get_sorted_frames(aug_masked_folder)

                # Create a new DatasetVideo for the augmented version
                augmented_video = DatasetVideo.__new__(DatasetVideo)
                augmented_video.raw = RawVideo(path=raw_video_path)
                augmented_video.masked = MaskedVideo(path=dataset_video.masked.path)
                augmented_video.raw.frames = augmented_raw_frames
                augmented_video.masked.frames = augmented_masked_frames
                augmented_video.output_dir = self.temp_dir
                augmented_video.name = f"{dataset_video.name}{aug.suffix}"

                # Add to dataset
                last_frame = max([rng.stop for rng in self.subject_to_frames.values()], default=0)
                # Extend the subject's frame range to include augmented frames
                original_range = self.subject_to_frames[subject_id]
                self.subject_to_frames[subject_id] = range(original_range.start, last_frame + len(augmented_video))
                self.videos.append(augmented_video)

            except Exception as e:
                logging.error(f"Failed to apply enrichment augmentation '{aug_name}': {e}")

    def _extract_subject_id(self, video_name: str) -> str:
        """Extract subject ID from video filename"""
        match = re.match(r"recordings_(?P<subject_id>\d+).*mp4", video_name)
        if match:
            return match.group("subject_id")
        return None

    def __len__(self):
        return sum(len(video) for video in self.videos)

    def __getitem__(self, index):
        def get_video_and_frame(index):
            """Get the video and frame index corresponding to the global index"""
            if index < 0 or index >= len(self):
                raise IndexError("Index out of range")

            video_index = 0
            # Iterate through videos to find the correct one
            while index >= len(self.videos[video_index]):
                index -= len(self.videos[video_index])
                video_index += 1
            return video_index, index

        video_index, frame_index = get_video_and_frame(index)
        video = self.videos[video_index]

        img_matrix = video.raw.get_frame_matrix(frame_index)
        mask_matrix = video.masked.get_frame_matrix(frame_index)

        # Apply in-place augmentations (on-the-fly during loading)
        if self.in_place_aug_objects:
            # Convert tensors back to numpy for augmentation
            # img_matrix expected shape: (1, H, W)
            img_np = img_matrix.squeeze(0).numpy()  # (H, W)

            # mask_matrix should be an index matrix (H, W) already in many cases; handle both
            mask_np = mask_matrix.numpy()

            # Convert to uint8 for OpenCV operations (raw frames are normalized floats)
            img_np = (img_np * 255).astype(np.uint8)

            for aug in self.in_place_aug_objects:
                if isinstance(aug, PairAugmentation):
                    # For pair augmentations, we need mask as uint8 image
                    mask_img = mask_np.astype(np.uint8)
                    img_np, mask_img = aug.apply_pair(img_np, mask_img)
                    mask_np = mask_img
                else:
                    # Apply augmentation based on supported types
                    if RawVideo in aug.supported_types:
                        img_np = aug.apply(img_np)

                    if MaskedVideo in aug.supported_types:
                        mask_np = aug.apply(mask_np)

            # Convert back to tensors
            img_matrix = torch.tensor(img_np.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
            mask_matrix = torch.tensor(mask_np, dtype=torch.long)

        return img_matrix, mask_matrix
    
    def get_subject_ids(self):
        """Get list of all subject IDs"""
        return list(self.subject_to_frames.keys())
    
    def get_video_indices_for_subjects(self, subject_ids):
        """Get all frame indices for given subject IDs"""
        indices = []
        for subject_id in subject_ids:
            if subject_id in self.subject_to_frames:
                indices.extend(self.subject_to_frames[subject_id])
        return indices
