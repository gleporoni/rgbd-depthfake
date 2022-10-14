from typing import Any

import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional as F


class RGB(pl.LightningModule):
    def __init__(self, conf, model_name) -> None:
        super().__init__()
        self.conf = conf

        if model_name == "resnet50":
            self.model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=True
            )
        elif model_name == "mobilenet_v2":
            torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
        elif model_name == "inception_v3":
            torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
        elif model_name == "efficientnet_b2":
            torch.hub.load("pytorch/vision:v0.10.0", "efficientnet_b2", pretrained=True)
        elif model_name == "shufflenet_v2_x1_0":
            torch.hub.load(
                "pytorch/vision:v0.10.0", "shufflenet_v2_x1_0", pretrained=True
            )
        elif model_name == "vit_h_14":
            torch.hub.load("pytorch/vision:v0.10.0", "vit_h_14", pretrained=True)
        else:
            raise NotImplementedError
        self.save_hyperparameters(conf)

    def forward(self, x) -> dict:
        x = self.model(x)
        return x

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.conf.model.learning_rate)
