from typing import Any

import timm
import pytorch_lightning as pl
import torchmetrics
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import models


class RGB(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.num_classes = self.conf.data.num_classes

        if self.conf.model.backbone == "resnet50":
            # init a pretrained resnet
            self.model = timm.create_model(
                "resnet50", pretrained=True, num_classes=self.num_classes
            )
        elif self.conf.model.backbone == "mobilenet_v2":
            self.model = timm.create_model(
                "mobilenetv2_100", pretrained=True, num_classes=self.num_classes
            )
        elif self.conf.model.backbone == "efficientnet_b2":
            self.model = timm.create_model(
                "efficientnet_b2", pretrained=True, num_classes=self.num_classes
            )
        elif self.conf.model.backbone == "shufflenet_v2_x1_0":
            self.model = models.shufflenet_v2_x1_0(pretrained=True)
            num_filters = self.model.fc.in_features

            # add a new classifier
            self.model.fc = torch.nn.Linear(num_filters, self.num_classes)
        elif self.conf.model.backbone == "xception":
            self.model = timm.create_model(
                "xception", pretrained=True, num_classes=self.num_classes
            )
        elif self.conf.model.backbone == "vit_base_patch16_224":
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=self.num_classes
            )

        else:
            raise NotImplementedError

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters(conf)

    def forward(self, x) -> dict:
        x = self.model(x)
        return x

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss)

        y_hat = y_hat.softmax(dim=-1)
        self.train_accuracy(y_hat, y)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        y_hat = y_hat.softmax(dim=-1)
        self.val_accuracy(y_hat, y)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True)

        return loss

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)

        y_hat = y_hat.softmax(dim=-1)
        self.test_accuracy(y_hat, y)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.conf.model.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
