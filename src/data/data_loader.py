from typing import Any, Union, List, Optional

from omegaconf import DictConfig
import logging

import torch
import torchvision.transforms as transforms
import torchvision.io as io
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.faceforensics import FaceForensics
from random import random


class FaceForensicsPlusPlus(pl.LightningDataModule):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf

        transform = (
            [transforms.ToTensor()]
            if self.conf.data.use_depth
            else [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.conf.data.mean, std=self.conf.data.std),
            ]
        )

        self.transform = transforms.Compose(transform)
        self.batch_size = self.conf.data.batch_size
        self.augmentation = transforms.Compose(
            [
                transforms.Lambda(self._compress_tensor),
                transforms.ConvertImageDtype(dtype=torch.float32),
                transforms.GaussianBlur(kernel_size=5, sigma = (4.0, 8.0))
            ]
        )
        self.quality = 75


    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_data = FaceForensics(
                conf=self.conf, split="train", transform=self.transform
            )
            self.val_data = FaceForensics(
                conf=self.conf, split="val", transform=self.transform
            )
        else:
            self.test_data = FaceForensics(
                conf=self.conf, split="test", transform=self.transform
            )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.conf.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.conf.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.conf.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )


    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.conf.data.augmentation and self.trainer.training:
            for i in range(batch['image'].shape[0]):
                if random() < 0.25:
                    batch['image'][i] = self.augmentation(batch['image'][i])
        return batch


    def _compress_tensor(self, tensor) -> torch.Tensor:
        # Check input shape
        if len(tensor.shape) == 3:
            channels, height, width = tensor.shape
        elif len(tensor.shape) == 2:
            height, width = tensor.shape
            channels = 1
            tensor = tensor.unsqueeze(-1)
        else:
            raise ValueError('Input tensor should have shape (height, width, channels) or (channels, height, width).')

        # Apply JPEG compression to each channel
        input_tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
        compressed_tensor = torch.zeros_like(input_tensor, dtype=torch.uint8)
        for j in range(channels):
            # Compress using JPEG
            compressed_data = io.encode_jpeg(input_tensor[None, j, :, :], quality=self.quality)
            # Decompress using JPEG
            decompressed_data = io.decode_jpeg(compressed_data)
            # Store decompressed data in compressed tensor
            compressed_tensor[j, :, :] = decompressed_data.squeeze(0)

        return compressed_tensor
