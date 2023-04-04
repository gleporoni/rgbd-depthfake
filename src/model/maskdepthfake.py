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
        if self.conf.data.use_hha:
            in_features = 3


        if self.conf.model.backbone == "xception":
            self.model = timm.create_model(
                "xception", pretrained=True, num_classes=self.num_classes
            )

            #load pretrained weights
            if self.conf.run.double_depth_network.use_pretain:
                print("not implemented")                
                # self.rgb_model = RGB(conf)
                # self.depth_model = RGB(conf)
                
                # base_path = Path(Path(__file__).parent, "../../")
                # rgb_checkpoint = Path(
                #     base_path,
                #     self.conf.run.double_depth_network.rgb_checkpoint,
                # )
                # depth_checkpoint = Path(
                #     base_path,
                #     self.conf.run.double_depth_network.depth_checkpoint,
                # )
                
                # #load weights for rgb part
                # self.rgb_model = self.rgb_model.load_from_checkpoint(checkpoint_path=rgb_checkpoint)


                # self.depth_model = self.depth_model.load_from_checkpoint(checkpoint_path=depth_checkpoint)

                # #remove rgb input and take only depth input in the first layer
                # # weights = self.depth_model.model.conv1.weight
                # # kernel_size = tuple(self.depth_model.model.conv1.kernel_size)
                # # stride = tuple(self.depth_model.model.conv1.stride)
                # # out_features = weights.shape[0]
                # # new_features = torch.nn.Conv2d(
                # #     in_features, out_features, kernel_size=kernel_size, stride=stride
                # # )
                # # new_features.weight = torch.nn.Parameter(weights.data[:, 3:, :, :])
                # # self.depth_model.model.conv1 = new_features

                # #freeze the models
                # for param in self.rgb_model.model.parameters():
                #     param.requires_grad = False
                # for param in self.depth_model.model.parameters():
                #     param.requires_grad = False
                
                # split = self.conf.run.double_depth_network.split
                # self.rgb_model = torch.nn.Sequential(*(list(self.rgb_model.model.children())[:split]))
                # self.depth_model = torch.nn.Sequential(*(list(self.depth_model.model.children())[:split]))
                # self.concat_layer = torch.nn.Conv2d(
                #     1456, 728, kernel_size=(1,1), stride=(1,1), padding = (0,0)
                # )
                # torch.nn.init.xavier_uniform_(self.concat_layer.weight)
                # self.model = torch.nn.Sequential(*(list(self.model.children())[split:])) 
                
            else:
                self.rgb_model = RGB(conf)
                self.depth_model = DepthFake(conf)


                #remove rgb input and take only depth input in the first layer
                weights = self.depth_model.model.conv1.weight
                kernel_size = tuple(self.depth_model.model.conv1.kernel_size)
                stride = tuple(self.depth_model.model.conv1.stride)
                out_features = weights.shape[0]
                new_features = torch.nn.Conv2d(
                    in_features, out_features, kernel_size=kernel_size, stride=stride
                )
                torch.nn.init.xavier_uniform_(new_features.weight)
                self.depth_model.model.conv1 = new_features



                # split = self.conf.run.double_depth_network.split
                split = 9
                self.rgb_model = copy.deepcopy(torch.nn.Sequential(*(list(self.rgb_model.model.children())[:split])))
                self.depth_model = copy.deepcopy(torch.nn.Sequential(*(list(self.depth_model.model.children())[:split])))


                # self.concat_layer = torch.nn.Conv2d(
                #     1456, 728, kernel_size=(1,1), stride=(1,1), padding = (0,0)
                # )

                # self.theta = torch.nn.Parameter(torch.zeros(728))
                # self.ones = torch.ones(2048).to('cuda')
                


                # self.fc_layer = torch.nn.Linear(in_features = 2048, out_features = 2, bias = True)
                # torch.nn.init.xavier_uniform_(self.concat_layer.weight)
                # torch.nn.init.xavier_uniform_(self.fc_layer.weight)
                self.rgb_model_final = copy.deepcopy(torch.nn.Sequential(*(list(self.model.children())[split:-1]))) 
                self.depth_model_final = copy.deepcopy(torch.nn.Sequential(*(list(self.model.children())[split:-1]))) 
                self.fc_layer = torch.nn.Linear(in_features = 4096, out_features = 2, bias = True)
                torch.nn.init.xavier_uniform_(self.fc_layer.weight)
                # self.model = None


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
        
        x_rgb = self.rgb_model_final(x_rgb)
        x_depth = self.depth_model_final(x_depth)

        x = torch.cat((x_rgb, x_depth), dim = 1)

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
