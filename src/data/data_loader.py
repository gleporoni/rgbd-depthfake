from typing import Any, Union, List, Optional

from omegaconf import DictConfig
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.faceforensics import FaceForensics


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
