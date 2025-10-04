import os
from torch.utils.data import Dataset
from pathlib import Path
import logging

from utils.video_utils import RawVideo, MaskedVideo


logging.basicConfig(level=logging.INFO)


class DatasetVideo:
    def __init__(self, raw_video: Path, masked_video: Path, output_dir: Path):
        self.raw = RawVideo(path=raw_video)
        self.masked = MaskedVideo(path=masked_video)
        self.output_dir = output_dir / raw_video.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split_to_frames(self):
        raw_folder = self.output_dir / "raw"
        masked_folder = self.output_dir / "masked"
        raw_folder.mkdir(parents=True, exist_ok=True)
        masked_folder.mkdir(parents=True, exist_ok=True)

        self.raw.split_to_frames(raw_folder)
        try:
            self.masked.split_to_frames(masked_folder)
        except Exception as e:
            logging.error(f"Error processing masked video {self.masked.path}: {e}")
            logging.error("removing created frames.")
            # Cleaning up
            os.rmdir(self.output_dir)
            self.raw.frames.clear()
            self.masked.frames.clear()

    def __len__(self):
        return len(self.raw.frames)
    
    def __getitem__(self, index):
        return self.raw[index], self.masked[index]


class USSegmentationDataset(Dataset):
    def __init__(self, raw_video_dir: str, masked_video_dir: str, temp_dir: str = "./temp_frames"):
        """
        Dataset for ultrasound segmentation from video files.
        
        Args:
            raw_video_dir: Directory containing raw video files
            masked_video_dir: Directory containing masked video files
            temp_dir: Temporary directory to store extracted frames
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.videos: list[DatasetVideo] = []
        
        # Process videos to extract frames
        self._process_videos(raw_video_dir, masked_video_dir)

        logging.info(f"Dataset initialized with {len(self.videos)} video pairs")

    def _process_videos(self, raw_video_dir: str, masked_video_dir: str):
        """Process video files to extract frames"""
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
                                            output_dir=self.temp_dir)
                dataset_video.split_to_frames()
                self.videos.append(dataset_video)
            except Exception:
                logging.exception(f"Error processing video {video_name}, skipping...")

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
        
        print(f"Loading frame {frame_index} from video {video_index}")
        img_matrix = video.raw.get_frame_matrix(frame_index)
        print(f"  Loaded raw frame: {img_matrix.shape}")
        mask_matrix = video.masked.get_frame_matrix(frame_index)
        print(f"  Loaded mask frame: {mask_matrix.shape}")
        
        return img_matrix, mask_matrix
