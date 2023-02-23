from typing import Any

import timm
import pytorch_lightning as pl
import torchmetrics
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import models


class DoubleDepthFake(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.num_classes = self.conf.data.num_classes


        if self.conf.model.backbone == "xception":
            self.model = timm.create_model(
                "xception", pretrained=True, num_classes=self.num_classes
            )

            # 9 for middle flow, 17 for exit flow

            self.model_RGB = torch.nn.Sequential(*(list(self.model.children())[:9])) 

            # Get the pre-trained weights of the first layer
            weights = self.model.conv1.weight
            kernel_size = tuple(self.model.conv1.kernel_size)
            stride = tuple(self.model.conv1.stride)
            out_features = weights.shape[0]
            new_features = torch.nn.Conv2d(
                1, out_features, kernel_size=kernel_size, stride=stride
            )
            # For Depth-channel weight should randomly initialized with Gaussian
            torch.nn.init.xavier_uniform_(new_features.weight)
            self.model.conv1 = new_features
            self.model_Depth = torch.nn.Sequential(*(list(self.model.children())[:9])) 


            # # setup concatenation layer

            # weights = self.model.block4.rep[1].conv1.weight
            # kernel_size = tuple(self.model.block4.rep[1].conv1.kernel_size)
            # stride = tuple(self.model.block4.rep[1].conv1.stride)
            # padding = tuple(self.model.block4.rep[1].conv1.padding)
            # groups = self.model.block4.rep[1].conv1.groups
            # bias = self.model.block4.rep[1].conv1.bias
            # out_features = weights.shape[0]
            # new_features = torch.nn.Conv2d(
            #     1456, out_features*2, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups*2, bias=bias
            # )

            # print(new_features.weight.shape)
            # print(weights.shape)
            # new_features.weight.data[:728, :, :, :] = torch.nn.Parameter(weights)
            # new_features.weight.data[728:, :, :, :] = torch.nn.Parameter(weights)
            # self.model.block4.rep[1].conv1 = new_features


            # weights = self.model.block4.rep[1].pointwise.weight
            # kernel_size = tuple(self.model.block4.rep[1].pointwise.kernel_size)
            # stride = tuple(self.model.block4.rep[1].pointwise.stride)
            # bias = self.model.block4.rep[1].pointwise.bias
            # out_features = weights.shape[0]
            # new_features = torch.nn.Conv2d(
            #     1456, out_features, kernel_size=kernel_size, stride=stride, bias=bias
            # )

            # print(new_features.weight.shape)
            # print(weights.shape)
            # new_features.weight.data[:, :728, :, :] = torch.nn.Parameter(weights)
            # new_features.weight.data[:, 728:, :, :] = torch.nn.Parameter(weights)
            # self.model.block4.rep[1].pointwise = new_features

            self.concat_layer = torch.nn.Conv2d(
                1456, 728, kernel_size=(3,3), stride=(1,1)
            )
            torch.nn.init.xavier_uniform_(self.concat_layer.weight)

            self.model = torch.nn.Sequential(*(list(self.model.children())[9:])) 


        else:
            raise NotImplementedError

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters(conf)

    def forward(self, x) -> dict:
        x_rgb = self.model_RGB(x[:,:3,:,:])
        x_depth = self.model_Depth(x[:,3:,:,:])

        x = torch.cat((x_rgb, x_depth), dim = 1)

        x = self.concat_layer(x)
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
