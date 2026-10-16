import os

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from vztbench.utils.image_tools import crop_maximal_rectangle, cv2_to_np

REAL_STATE_NAMES = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
)


class RealImageDataset(Dataset):
    """Raw images with the same preprocessing as in scripts/encode_dataset.py."""

    def __init__(self, image_dir: str, image_size: int):
        self.image_dir = image_dir
        self.image_size = image_size
        self.files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        returns:
            image: [3, H, W], dtype = uint8, channel_order = RGB
        """
        image = cv2.imread(os.path.join(self.image_dir, self.files[idx]))
        image = crop_maximal_rectangle(
            image, target_height=self.image_size, target_width=self.image_size
        )
        image = cv2_to_np(image)
        return torch.from_numpy(image).permute(2, 0, 1)


class FIDEvaluator:
    """
    FID against the statistics of real images.
    Real statistics are computed once and cached at stats_path, so all runs share them.
    """

    def __init__(
        self,
        real_image_dir: str,
        stats_path: str,
        image_size: int = 256,
        device: str = "cuda",
        batch_size: int = 256,
        num_workers: int = 8,
    ):
        self.device = device
        self.metric = FrechetInceptionDistance(
            feature=2048, reset_real_features=False
        ).to(device)

        if os.path.exists(stats_path):
            self._load_real_statistics(stats_path)
        else:
            self._compute_real_statistics(
                RealImageDataset(real_image_dir, image_size),
                batch_size,
                num_workers,
            )
            self._save_real_statistics(stats_path)

    def _compute_real_statistics(self, dataset, batch_size, num_workers):
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        for images in tqdm(loader, desc="FID real statistics"):
            self.metric.update(images.to(self.device, non_blocking=True), real=True)

    def _save_real_statistics(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {name: getattr(self.metric, name).cpu() for name in REAL_STATE_NAMES}
        torch.save(state, path)

    def _load_real_statistics(self, path: str):
        state = torch.load(path, map_location=self.device)
        for name in REAL_STATE_NAMES:
            setattr(self.metric, name, state[name])

    def update(self, images: torch.Tensor):
        """
        args:
            images: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        images = images.clamp(0, 1).mul(255).round().to(torch.uint8)
        self.metric.update(images, real=False)

    def compute(self) -> float:
        """Returns FID of all images passed to update() and clears them."""
        fid = self.metric.compute().item()
        self.metric.reset()
        return fid
