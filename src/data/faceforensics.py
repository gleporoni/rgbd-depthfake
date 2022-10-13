import csv
import torch
import numpy as np
import pandas as pd
import glob

from random import randrange

from typing import Any
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
        self.rgb_path = Path(self.conf.data.rgb_path)
        self.depth_path = Path(self.conf.data.depth_path)
        self.compression_level = self.conf.data.compression_level
        self.real = self.conf.data.real
        self.attacks = self.conf.data.attacks

        self.dataset = self._load_data()

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        This function returns a tuple that is further passed to collate_fn
        """

        # Load the image and apply transformations
        image = Image.fromarray(np.ones((224, 224, 3)).astype("uint8"), "RGB")

        if self.transform:
            image = self.transform(image)
        label = 1

        return {
            "image": image,
            "label": label,
        }

    def _load_data(self, use_depth: bool = False, use_attacks: list = False):
        images = []
        labels = []
        for compression in self.compression_level:
            for r in self.real:
                list_of_images = [path for path in glob.glob(str(Path(self.rgb_path, "Real", compression, r))+"/*/")]
                print(list_of_images)
                exit(0)
                images.append(list_of_images)
                
                list_of_labels = [r for i in range(len(list_of_images))]
                labels.append(list_of_labels)

            if use_attacks:
                for a in self.attacks:
                    list_of_images = [path for path in Path(self.rgb_path, "Fake", compression, a).glob("*/*.jpg")]
                    images.append(list_of_images)
                    
                    list_of_labels = [a for i in range(len(list_of_images))]
                    labels.append(list_of_labels)

        print([(a, b) for (a,b) in zip(images, labels)])
        exit(0)