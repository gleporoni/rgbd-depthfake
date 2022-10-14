import csv
import torch
import numpy as np
import pandas as pd
import glob

from random import randrange

from typing import Any, Tuple
from pathlib import Path
from omegaconf import DictConfig
from PIL import Image

from torch.utils.data import Dataset


class FaceForensics(Dataset):
    """
    Dataset loader for T4SA dataset.
    """

    def __init__(
        self,
        conf: DictConfig,
        split: str,
        transform: Any = None,
    ):
        self.conf = conf
        self.split = split

        # Data dirs
        self.base_path = Path(Path(__file__).parent, "../../")
        self.rgb_path = Path(self.base_path, self.conf.data.rgb_path)
        self.depth_path = Path(self.base_path, self.conf.data.depth_path)

        # Dataset info
        self.compression_level = self.conf.data.compression_level
        self.real = self.conf.data.real
        self.attacks = self.conf.data.attacks

        self.dataset = self._load_data(
            use_attacks=self.conf.data.use_attacks, use_depth=self.conf.data.use_depth
        )

        # Check if the dataset is well constructed before training
        self._data_sanity_check(self.dataset)

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        This function returns a tuple that is further passed to collate_fn
        """

        # Load the image and apply transformations
        image = Image.fromarray(np.ones((224, 224, 3)).astype("uint8"), "RGB")
        depth = []

        if self.transform:
            image = self.transform(image)
        label = 1

        return {
            "image": image,
            "depth": depth,
            "label": label,
        }

    def _load_data(
        self, use_depth: bool = False, use_attacks: list = False
    ) -> pd.DataFrame:
        images = []
        depths = []
        labels = []

        # Loop over compression levels
        for compression in self.compression_level:
            # Loop over real videos
            for r in self.real:
                list_of_images = [
                    path
                    for path in Path(self.rgb_path, "Real", compression, r).glob(
                        "*/*.jpg"
                    )
                ]
                images += list_of_images

                list_of_labels = [r for i in range(len(list_of_images))]
                labels += list_of_labels

                if use_depth:
                    depths += self._load_depth(
                        compression=compression, label="Real", source=r
                    )

            if use_attacks:
                # Loop over the attacks
                for a in self.attacks:
                    list_of_images = [
                        path
                        for path in Path(self.rgb_path, "Fake", compression, a).glob(
                            "*/*.jpg"
                        )
                    ]
                    images += list_of_images

                    list_of_labels = [a for i in range(len(list_of_images))]
                    labels += list_of_labels

                    if use_depth:
                        depths += self._load_depth(
                            compression=compression, label="Fake", source=a
                        )

        dataset = pd.DataFrame(
            data={
                "images": images,
                "depths": depths,
                "labels": labels,
            }
        )

        return dataset

    def _load_depth(self, label: str, compression: str, source: str) -> list:
        assert (
            label in ("Real", "Fake")
            and source
            in (
                "actors",
                "youtube",
                "Deepfakes",
                "Face2Face",
                "FaceShifter",
                "FaceSwap",
                "NeuralTextures",
            )
            and compression in ("raw", "c24", "c40")
        )

        depths = [
            path
            for path in Path(self.depth_path, label, compression, source).glob(
                "*/*.jpg"
            )
        ]

        return depths

    def _data_sanity_check(self, dataset: pd.DataFrame) -> None:
        """
        Check if the dataset is well constructed.
        """
        for _, row in dataset.iterrows():
            if (
                row.images.match(
                    str(Path("*", row.depths.parent.name, row.depths.name))
                )
                == False
            ):
                raise ValueError(f"Non matching inputs:\n{row.images}{row.depths}")
