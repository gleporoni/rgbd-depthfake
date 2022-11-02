import csv
import torch
import numpy as np
import pandas as pd
import glob
import logging

from random import randrange

from typing import Any, Tuple
from pathlib import Path
from omegaconf import DictConfig
from PIL import Image

from torch.utils.data import Dataset

VAL_VIDEOS = ["720", "672", "939", "115", "284", "263", "402", "453", "820", "818", "762", "832", "834", "852", "922", "898", "104", "126", "106", "198", "159", "175", "416", "342", "857", "909", "599", "585", "443", "514", "566", "617", "472", "511", "325", "492", "816", "649", "583", "558", "933", "925", "419", "824", "465", "482", "565", "589", "261", "254", "992", "980", "157", "245", "571", "746", "947", "951", "926", "900", "493", "538", "468", "470", "915", "895", "362", "354", "440", "364", "640", "638", "827", "817", "793", "768", "837", "890", "004", "982", "192", "134", "745", "777", "299", "145", "742", "775", "586", "223", "483", "370", "779", "794", "971", "564", "273", "807", "991", "064", "664", "668", "823", "584", "656", "666", "557", "560", "471", "455", "042", "084", "979", "875", "316", "369", "091", "116", "023", "923", "702", "612", "904", "046", "647", "622", "958", "956", "606", "567", "632", "548", "927", "912", "350", "349", "595", "597", "727", "729"]
TEST_VIDEOS = ["953", "974", "012", "026", "078", "955", "623", "630", "919", "015", "367", "371", "847", "906", "529", "633", "418", "507", "227", "169", "389", "480", "821", "812", "670", "661", "158", "379", "423", "421", "352", "319", "579", "701", "488", "399", "695", "422", "288", "321", "705", "707", "306", "278", "865", "739", "995", "233", "755", "759", "467", "462", "314", "347", "741", "731", "970", "973", "634", "660", "494", "445", "706", "479", "186", "170", "176", "190", "380", "358", "214", "255", "454", "527", "425", "485", "388", "308", "384", "932", "035", "036", "257", "420", "924", "917", "114", "102", "732", "691", "550", "452", "280", "249", "842", "714", "625", "650", "024", "073", "044", "945", "896", "128", "862", "047", "607", "683", "517", "521", "682", "669", "138", "142", "552", "851", "376", "381", "000", "003", "048", "029", "724", "725", "608", "675", "386", "154", "220", "219", "801", "855", "161", "141", "949", "868", "880", "135", "429", "404"]


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
        self.num_classes = self.conf.data.num_classes

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
        if self.dataset.depths[0] is not None:
            self._data_sanity_check(self.dataset)

        # Convert string labels to categoricals
        self._to_categorical()

        # Split the dataset
        self._data_split()

        self.transform = transform
        self.log = logging.getLogger(__name__)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        This function returns a tuple that is further passed to collate_fn
        """
        # Load the image and apply transformations
        image = Image.open(self.dataset.images[idx]).convert("RGB")
        #depth = Image.fromarray(self.dataset.depths[idx].astype("uint8"), "RGB")

        if self.transform:
            image = self.transform(image)
        label = self.dataset.classes[idx]
        
        return {
            "image": image,
            #"depth": depth,
            "label": label,
        }

    def _load_data(
        self, use_depth: bool = False, use_attacks: list = False
    ) -> pd.DataFrame:
        """
        Load the RGB images.
        """
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

                list_of_labels = [r for _ in range(len(list_of_images))]
                labels += list_of_labels

                if use_depth:
                    depths += self._load_depth(
                        compression=compression, label="Real", source=r
                    )
                else:
                    for _ in range(len(list_of_images)):
                        depths.append(None)

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

                    list_of_labels = [a for _ in range(len(list_of_images))]
                    labels += list_of_labels

                    if use_depth:
                        depths += self._load_depth(
                            compression=compression, label="Fake", source=a
                        )
                    else:
                        for _ in range(len(list_of_images)):
                            depths.append(None)

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

    def _to_categorical(self):
        """
        Converts strings labels to integers.
        """
        classes = []
        for label in self.dataset.labels:
            if label in self.conf.data.real:
                classes.append(0)
            else:
                if self.num_classes == 2:
                    classes.append(1)
                else:
                    classes.append(list(self.conf.data.attacks).index(label)+1)

        self.dataset["classes"] = classes

    def _data_split(self):
        """
        Splits the dataset according to the actual split.
        """
        split = []
        for _, row in self.dataset.iterrows():
            video = str(row.images.parent).split("/")[-1].split("_")[0]
            if video in VAL_VIDEOS:
                split.append("val")
            elif video in TEST_VIDEOS:
                split.append("test")
            else:
                split.append("train")
        self.dataset["split"] = split

        self.dataset = self.dataset.loc[self.dataset['split'] == self.split].reset_index()