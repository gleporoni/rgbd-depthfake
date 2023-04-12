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
import copy



class MaskDepthFake(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.num_classes = self.conf.data.num_classes
        in_features = 1
        if self.conf.data.depth_type  == 'hha':
            in_features = 3

        self.classify_on = self.conf.model.classify_on
        

        if self.conf.model.backbone == "xception":
            self.model = timm.create_model(
                "xception", pretrained=True, num_classes=self.num_classes
            )

            self.rgb_model = RGB(conf)
            self.depth_model = RGB(conf)

            if in_features == 1:
                #remove rgb input and take only depth input in the first layer
                weights = self.depth_model.model.conv1.weight
                kernel_size = tuple(self.depth_model.model.conv1.kernel_size)
                stride = tuple(self.depth_model.model.conv1.stride)
                out_features = weights.shape[0]
                new_features = torch.nn.Conv2d(
                    1, out_features, kernel_size=kernel_size, stride=stride
                )
                torch.nn.init.xavier_uniform_(new_features.weight)
                self.depth_model.model.conv1 = new_features


            split = self.conf.model.split
            self.rgb_model = copy.deepcopy(torch.nn.Sequential(*(list(self.rgb_model.model.children())[:split])))
            self.depth_model = copy.deepcopy(torch.nn.Sequential(*(list(self.depth_model.model.children())[:split])))

            if self.classify_on == 'rgbd':
                self.rgb_model_final = copy.deepcopy(torch.nn.Sequential(*(list(self.model.children())[split:-1])))
                self.depth_model_final = copy.deepcopy(torch.nn.Sequential(*(list(self.model.children())[split:-1])))
                self.fc_layer = torch.nn.Linear(in_features = 4096, out_features = 2, bias = True)
                torch.nn.init.xavier_uniform_(self.fc_layer.weight)
            else:
                self.depth_model_final = copy.deepcopy(torch.nn.Sequential(*(list(self.model.children())[split:])))

            self.model = None                

        elif self.conf.model.backbone == "mobilenet_v2":
            self.model = timm.create_model(
                "mobilenetv2_100", pretrained=True, num_classes=self.num_classes
            )

            self.rgb_model = RGB(conf)
            self.depth_model = RGB(conf)
            
            if in_features == 1:
                weights = self.depth_model.model.conv_stem.weight
                kernel_size = tuple(self.depth_model.model.conv_stem.kernel_size)
                stride = tuple(self.depth_model.model.conv_stem.stride)
                out_features = weights.shape[0]
                new_features = torch.nn.Conv2d(
                    1, out_features, kernel_size=kernel_size, stride=stride
                )

                torch.nn.init.xavier_uniform_(new_features.weight)
                self.depth_model.model.conv_stem = new_features

            rgb_tmp = list((list(self.rgb_model.model.children())[2]).children())
            rgb_layer = list(self.rgb_model.model.children())[:2]
            rgb_layer.extend(rgb_tmp[:4])

            depth_tmp = list((list(self.depth_model.model.children())[2]).children())
            depth_layer = list(self.depth_model.model.children())[:2]
            depth_layer.extend(depth_tmp[:4])

            self.rgb_model = copy.deepcopy(torch.nn.Sequential(*rgb_layer))
            self.depth_model = copy.deepcopy(torch.nn.Sequential(*depth_layer))

            
            if self.classify_on == 'rgbd':

                rgb_layer = rgb_tmp[4:] 
                rgb_layer.extend(list(self.model.children())[3:-1])
                
                depth_layer = depth_tmp[4:]
                depth_layer.extend(list(self.model.children())[3:-1])

                self.rgb_model_final = copy.deepcopy(torch.nn.Sequential(*rgb_layer)) 
                
                self.fc_layer = torch.nn.Linear(in_features = 2560, out_features = 2, bias = True)
                torch.nn.init.xavier_uniform_(self.fc_layer.weight)

            else:
                depth_layer.extend(list(self.model.children())[3:])

            self.depth_model_final = copy.deepcopy(torch.nn.Sequential(*depth_layer)) 

            self.model = None

        else:
            raise NotImplementedError

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters(conf)

    def forward(self, x) -> dict:
        x_rgb = self.rgb_model(x[:,:3,:,:])
        x_depth = self.depth_model(x[:,3:,:,:])
        
        x_mask = x_rgb.gt(0).to(torch.float32)

        x_depth = x_mask * x_depth

        x = self.depth_model_final(x_depth)

        if self.classify_on == 'rgbd': 

            x_rgb = self.rgb_model_final(x_rgb)
            x = torch.cat((x_rgb, x), dim = 1)
            x = self.fc_layer(x)

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
