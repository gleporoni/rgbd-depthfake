from typing import Any

import timm
import pytorch_lightning as pl
import torchmetrics
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import models
from pathlib import Path
from model.rgb import RGB
from model.depthfake import DepthFake



class AttentionDepthFake(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.num_classes = self.conf.data.num_classes


        if self.conf.model.backbone == "resnet50":
            model = timm.create_model(
                "resnet50", pretrained=True, num_classes=self.num_classes
            )

            # rgb branch

            self.conv_rgb =  torch.nn.Sequential(*(list(model.children())[:3]))
            self.maxpool_rgb = model.maxpool
            self.layer1_rgb = model.layer1
            self.layer2_rgb = model.layer2
            self.layer3_rgb = model.layer3
            self.layer4_rgb = model.layer4



            # depth branch

            tmp_conv_depth = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
            tmp_layer_depth = list(model.children())[1:3]
            tmp_layer_depth.insert(0, tmp_conv_depth)
            self.conv_depth = torch.nn.Sequential(*(tmp_layer_depth))
            self.maxpool_detph = model.maxpool

            self.layer1_depth = model.layer1
            self.layer2_depth = model.layer2
            self.layer3_depth = model.layer3
            self.layer4_depth = model.layer4

            # mixed branch

            self.atten_rgb_0 = self.channel_attention(64)
            self.atten_depth_0 = self.channel_attention(64)

            self.maxpool_mix = model.maxpool

            self.atten_rgb_1 = self.channel_attention(64*4)
            self.atten_depth_1 = self.channel_attention(64*4)
            self.atten_rgb_2 = self.channel_attention(128*4)
            self.atten_depth_2 = self.channel_attention(128*4)
            self.atten_rgb_3 = self.channel_attention(256*4)
            self.atten_depth_3 = self.channel_attention(256*4)
            self.atten_rgb_4 = self.channel_attention(512*4)
            self.atten_depth_4 = self.channel_attention(512*4)

            self.layer1_mix = model.layer1
            self.layer2_mix = model.layer2
            self.layer3_mix = model.layer3
            self.layer4_mix = model.layer4


            # classification

            self.global_pool = model.global_pool
            self.fc = model.fc

            tmp_conv_depth = None
            tmp_layer_depth = None
            model = None

        else:
            raise NotImplementedError

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters(conf)

    def channel_attention(self, num_channel, ablation=False):
        pool = torch.nn.AdaptiveAvgPool2d(1)
        conv = torch.nn.Conv2d(num_channel, num_channel, kernel_size=1)
        activation = torch.nn.Sigmoid()

        return torch.nn.Sequential(*[pool, conv, activation])

    def forward(self, x) -> dict:
        x_rgb = x[:,:3,:,:]
        x_depth = x[:,3:,:,:]

        rgb = self.conv_rgb(x_rgb)
        depth = self.conv_depth(x_depth)
        atten_rgb = self.atten_rgb_0(rgb)
        atten_depth = self.atten_depth_0(depth)
        m0 = rgb.mul(atten_rgb) + depth.mul(atten_depth)

        rgb = self.maxpool_rgb(rgb)
        depth = self.maxpool_detph(depth)
        m = self.maxpool_mix(m0)

        # block 1
        rgb = self.layer1_rgb(rgb)
        depth = self.layer1_depth(depth)
        m = self.layer1_mix(m)

        atten_rgb = self.atten_rgb_1(rgb)
        atten_depth = self.atten_depth_1(depth)
        m1 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

        # block 2
        rgb = self.layer2_rgb(rgb)
        depth = self.layer2_depth(depth)
        m = self.layer2_mix(m1)

        atten_rgb = self.atten_rgb_2(rgb)
        atten_depth = self.atten_depth_2(depth)
        m2 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

        # block 3
        rgb = self.layer3_rgb(rgb)
        depth = self.layer3_depth(depth)
        m = self.layer3_mix(m2)

        atten_rgb = self.atten_rgb_3(rgb)
        atten_depth = self.atten_depth_3(depth)
        m3 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

        # block 4
        rgb = self.layer4_rgb(rgb)
        depth = self.layer4_depth(depth)
        m = self.layer4_mix(m3)

        atten_rgb = self.atten_rgb_4(rgb)
        atten_depth = self.atten_depth_4(depth)
        m4 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

        x = self.global_pool(m4)
        x = self.fc(x)

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
