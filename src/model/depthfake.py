from typing import Any

import timm
import pytorch_lightning as pl
import torchmetrics
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import models


class DepthFake(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.num_classes = self.conf.data.num_classes
        in_features = 4
        if self.conf.data.input_type == 'rgbd' and self.conf.data.depth_type  == 'hha':
            in_features = 6
        elif self.conf.data.input_type == 'd':
            in_features = 1
            if self.conf.data.depth_type  == 'hha':
                in_features = 3

        if self.conf.model.backbone == "resnet50":
            # init a pretrained resnet
            self.model = timm.create_model(
                "resnet50", pretrained=True, num_classes=self.num_classes
            )
            # Get the pre-trained weights of the first layer
            weights = self.model.conv1.weight
            kernel_size = tuple(self.model.conv1.kernel_size)
            stride = tuple(self.model.conv1.stride)
            out_features = weights.shape[0]
            new_features = torch.nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride
            )
            # For Depth-channel weight should randomly initialized with Gaussian
            torch.nn.init.xavier_uniform_(new_features.weight)
            # For RGB it should be copied from pretrained weights
            if in_features > 2:
                new_features.weight.data[:, :3, :, :] = torch.nn.Parameter(weights)
            # if self.conf.data.use_hha:
                # new_features.weight.data[:, 3:, :, :] = torch.nn.Parameter(weights)
            # Update the pre-trained weights of the first layer
            self.model.conv1 = new_features
        elif self.conf.model.backbone == "mobilenet_v2":
            self.model = timm.create_model(
                "mobilenetv2_100", pretrained=True, num_classes=self.num_classes
            )
            # Get the pre-trained weights of the first layer
            weights = self.model.conv_stem.weight
            kernel_size = tuple(self.model.conv_stem.kernel_size)
            stride = tuple(self.model.conv_stem.stride)
            out_features = weights.shape[0]
            new_features = torch.nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride
            )
            # For Depth-channel weight should randomly initialized with Gaussian
            torch.nn.init.xavier_uniform_(new_features.weight)
            # For RGB it should be copied from pretrained weights
            if in_features > 2:
                new_features.weight.data[:, :3, :, :] = torch.nn.Parameter(weights)
            # if self.conf.data.use_hha:
                # new_features.weight.data[:, 3:, :, :] = torch.nn.Parameter(weights)
            # Update the pre-trained weights of the first layer
            self.model.conv_stem = new_features
        elif self.conf.model.backbone == "efficientnet_b2":
            self.model = timm.create_model(
                "efficientnet_b2", pretrained=True, num_classes=self.num_classes
            )
            # Get the pre-trained weights of the first layer
            weights = self.model.conv_stem.weight
            kernel_size = tuple(self.model.conv_stem.kernel_size)
            stride = tuple(self.model.conv_stem.stride)
            out_features = weights.shape[0]
            new_features = torch.nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride
            )
            # For Depth-channel weight should randomly initialized with Gaussian
            torch.nn.init.xavier_uniform_(new_features.weight)
            # For RGB it should be copied from pretrained weights
            if in_features > 2:
                new_features.weight.data[:, :3, :, :] = torch.nn.Parameter(weights)
            # if self.conf.data.use_hha:
                # new_features.weight.data[:, 3:, :, :] = torch.nn.Parameter(weights)
            # Update the pre-trained weights of the first layer
            self.model.conv_stem = new_features
        elif self.conf.model.backbone == "shufflenet_v2_x1_0":
            self.model = models.shufflenet_v2_x1_0(pretrained=True)
            num_filters = self.model.fc.in_features

            # add a new classifier
            self.model.fc = torch.nn.Linear(num_filters, self.num_classes)

            # Get the pre-trained weights of the first layer
            weights = self.model.conv1[0].weight
            kernel_size = tuple(self.model.conv1[0].kernel_size)
            stride = tuple(self.model.conv1[0].stride)
            out_features = weights.shape[0]
            new_features = torch.nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride
            )
            # For Depth-channel weight should randomly initialized with Gaussian
            torch.nn.init.xavier_uniform_(new_features.weight)
            # For RGB it should be copied from pretrained weights
            if in_features > 2:
                new_features.weight.data[:, :3, :, :] = torch.nn.Parameter(weights)
            # if self.conf.data.use_hha:
                # new_features.weight.data[:, 3:, :, :] = torch.nn.Parameter(weights)
            # Update the pre-trained weights of the first layer
            self.model.conv1[0] = new_features
        elif self.conf.model.backbone == "xception":
            self.model = timm.create_model(
                "xception", pretrained=True, num_classes=self.num_classes
            )
            # Get the pre-trained weights of the first layer
            weights = self.model.conv1.weight
            kernel_size = tuple(self.model.conv1.kernel_size)
            stride = tuple(self.model.conv1.stride)
            out_features = weights.shape[0]
            new_features = torch.nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride
            )
            # For Depth-channel weight should randomly initialized with Gaussian
            torch.nn.init.xavier_uniform_(new_features.weight)
            # For RGB it should be copied from pretrained weights
            if in_features > 2:
                new_features.weight.data[:, :3, :, :] = torch.nn.Parameter(weights)
            # if self.conf.data.use_hha:
                # new_features.weight.data[:, 3:, :, :] = torch.nn.Parameter(weights)
            # Update the pre-trained weights of the first layer
            self.model.conv1 = new_features
        elif self.conf.model.backbone == "vit_base_patch16_224":
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=self.num_classes
            )
            # Get the pre-trained weights of the first layer
            weights = self.model.patch_embed.proj.weight
            kernel_size = tuple(self.model.patch_embed.proj.kernel_size)
            stride = tuple(self.model.patch_embed.proj.stride)
            out_features = weights.shape[0]
            new_features = torch.nn.Conv2d(
                in_features, out_features, kernel_size=kernel_size, stride=stride
            )
            # For Depth-channel weight should randomly initialized with Gaussian
            torch.nn.init.xavier_uniform_(new_features.weight)
            # For RGB it should be copied from pretrained weights
            if in_features > 2:
                new_features.weight.data[:, :3, :, :] = torch.nn.Parameter(weights)
            # if self.conf.data.use_hha:
                # new_features.weight.data[:, 3:, :, :] = torch.nn.Parameter(weights)
            # Update the pre-trained weights of the first layer
            self.model.patch_embed.proj = new_features
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
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.conf.model.learning_rate,
            weight_decay=self.conf.model.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
