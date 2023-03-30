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






class DoubleDepthFakeMask(pl.LightningModule):
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
                
                self.rgb_model = RGB(conf)
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
                # self.rgb_model = RGB(conf)
                tmp_model = list(self.model.children())

                # net before block 1
                self.rgb_model_init = copy.deepcopy(torch.nn.Sequential(*(tmp_model[:9])))
                self.depth_model_init = copy.deepcopy(torch.nn.Sequential(*(tmp_model[:9])))

                #remove rgb input and take only depth input in the first layer
                # weights = self.depth_model_init[0].weight
                # kernel_size = tuple(self.depth_model_init[0].kernel_size)
                # stride = tuple(self.depth_model_init[0].stride)
                # out_features = weights.shape[0]
                # new_features = torch.nn.Conv2d(
                #     in_features, out_features, kernel_size=kernel_size, stride=stride
                # )
                # # new_features.weight = torch.nn.Parameter(weights.data[:, 3:, :, :])
                # torch.nn.init.xavier_uniform_(new_features.weight)
                # self.depth_model_init[0] = new_features


                # block 4
                self.rgb_conv_block_4 = copy.deepcopy(list(tmp_model[9].children())[0])
                self.depth_conv_block_4 = copy.deepcopy(list(tmp_model[9].children())[0])
                self.concat_block_4 = copy.deepcopy(list(tmp_model[9].children())[0])                
                self.theta_block_4 = torch.nn.Parameter(0.5)

                # block 5
                self.rgb_conv_block_5 = copy.deepcopy(list(tmp_model[10].children())[0])
                self.depth_conv_block_5 = copy.deepcopy(list(tmp_model[10].children())[0])
                self.concat_block_5 = copy.deepcopy(list(tmp_model[10].children())[0])                
                self.theta_block_5 = torch.nn.Parameter(0.5)

                # block 6
                self.rgb_conv_block_6 = copy.deepcopy(list(tmp_model[11].children())[0])
                self.depth_conv_block_6 = copy.deepcopy(list(tmp_model[11].children())[0])
                self.concat_block_6 = copy.deepcopy(list(tmp_model[11].children())[0])
                self.theta_block_6 = torch.nn.Parameter(0.5)


                # block 7
                self.rgb_conv_block_7 = copy.deepcopy(list(tmp_model[12].children())[0])
                self.depth_conv_block_7 = copy.deepcopy(list(tmp_model[12].children())[0])
                self.concat_block_7 = copy.deepcopy(list(tmp_model[12].children())[0])
                self.theta_block_7 = torch.nn.Parameter(0.5)

                # block 8
                self.rgb_conv_block_8 = copy.deepcopy(list(tmp_model[13].children())[0])
                self.depth_conv_block_8 = copy.deepcopy(list(tmp_model[13].children())[0])
                self.concat_block_8 = copy.deepcopy(list(tmp_model[13].children())[0])
                self.theta_block_8 = torch.nn.Parameter(0.5)

                # block 9
                self.rgb_conv_block_9 = copy.deepcopy(list(tmp_model[14].children())[0])
                self.depth_conv_block_9 = copy.deepcopy(list(tmp_model[14].children())[0])
                self.concat_block_9 = copy.deepcopy(list(tmp_model[14].children())[0])
                self.theta_block_9 = torch.nn.Parameter(0.5)

                # block 10
                self.rgb_conv_block_10 = copy.deepcopy(list(tmp_model[15].children())[0])
                self.depth_conv_block_10 = copy.deepcopy(list(tmp_model[15].children())[0])
                self.concat_block_10 = copy.deepcopy(list(tmp_model[15].children())[0])
                self.theta_block_10 = torch.nn.Parameter(0.5)

                # block 11
                self.rgb_conv_block_11 = copy.deepcopy(list(tmp_model[16].children())[0])
                self.depth_conv_block_11 = copy.deepcopy(list(tmp_model[16].children())[0])
                self.concat_block_11 = copy.deepcopy(list(tmp_model[16].children())[0])
                self.theta_block_11 = torch.nn.Parameter(0.5)

                # net after block 11
                self.concat_model_final = copy.deepcopy(torch.nn.Sequential(*(tmp_model[17:25])))
                self.fc_layer = torch.nn.Linear(in_features = 2048, out_features = 2, bias = True)

                self.model = None

                # self.depth_model = DepthFake(conf)


                # #remove rgb input and take only depth input in the first layer
                # weights = self.depth_model.model.conv1.weight
                # kernel_size = tuple(self.depth_model.model.conv1.kernel_size)
                # stride = tuple(self.depth_model.model.conv1.stride)
                # out_features = weights.shape[0]
                # new_features = torch.nn.Conv2d(
                #     in_features, out_features, kernel_size=kernel_size, stride=stride
                # )
                # new_features.weight = torch.nn.Parameter(weights.data[:, 3:, :, :])
                # self.depth_model.model.conv1 = new_features

                # # split = self.conf.run.double_depth_network.split

                # # n = sum(1 for _ in self.rgb_model.model.parameters())
                # # half = int(n/2)

                # # for i, param in enumerate(self.rgb_model.model.parameters()):
                # #     if i > half:
                # #         break
                # #     param.requires_grad = False
                # # for i, param in enumerate(self.depth_model.model.parameters()):
                # #     if i > half:
                # #         break
                # #     param.requires_grad = False
                # self.rgb_model = torch.nn.Sequential(*(list(self.rgb_model.model.children())[:-1]))
                # self.depth_model = torch.nn.Sequential(*(list(self.depth_model.model.children())[:-1]))
                # # self.concat_layer = torch.nn.Conv2d(
                # #     1456, 728, kernel_size=(1,1), stride=(1,1), padding = (0,0)
                # # )

                # self.theta = torch.nn.Parameter(torch.ones(2048)*0.5)
                # self.ones = torch.ones(2048).to('cuda')

                # self.fc_layer = torch.nn.Linear(in_features = 2048, out_features = 2, bias = True)
                # # torch.nn.init.xavier_uniform_(self.concat_layer.weight)
                # torch.nn.init.xavier_uniform_(self.fc_layer.weight)
                # # self.model = torch.nn.Sequential(*(list(self.model.children())[split:])) 
                # self.model = None


        else:
            raise NotImplementedError

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters(conf)

    def forward(self, x) -> dict:
        x_rgb = self.rgb_model_init(x[:,:3,:,:])
        x_depth = self.depth_model_init(x[:,3:,:,:])
        
        # block 4
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_4*x_rgb + (1-self.theta_block_4)*x_depth
        
        x = x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_4(x)
        x_rgb = self.rgb_conv_block_4(x_rgb)
        x_depth = self.depth_conv_block_4(x_depth)

        # block 5
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_5*x_rgb + (1-self.theta_block_5)*x_depth
        
        x = x + x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_5(x)
        x_rgb = self.rgb_conv_block_5(x_rgb)
        x_depth = self.depth_conv_block_5(x_depth)

        # block 6
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_6*x_rgb + (1-self.theta_block_6)*x_depth
        
        x = x + x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_6(x)
        x_rgb = self.rgb_conv_block_6(x_rgb)
        x_depth = self.depth_conv_block_6(x_depth)

        # block 7
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_7*x_rgb + (1-self.theta_block_7)*x_depth
        
        x = x + x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_7(x)
        x_rgb = self.rgb_conv_block_7(x_rgb)
        x_depth = self.depth_conv_block_7(x_depth)

        # block 8
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_8*x_rgb + (1-self.theta_block_8)*x_depth
        
        x = x + x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_8(x)
        x_rgb = self.rgb_conv_block_8(x_rgb)
        x_depth = self.depth_conv_block_8(x_depth)

        # block 9
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_9*x_rgb + (1-self.theta_block_9)*x_depth
        
        x = x + x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_9(x)
        x_rgb = self.rgb_conv_block_9(x_rgb)
        x_depth = self.depth_conv_block_9(x_depth)

        # block 10
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_10*x_rgb + (1-self.theta_block_10)*x_depth
        
        x = x + x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_10(x)
        x_rgb = self.rgb_conv_block_10(x_rgb)
        x_depth = self.depth_conv_block_10(x_depth)

        # block 11
        x_rgb_mask = x_rgb.gt(0).to(torch.float32)
        x_depth_mask = x_depth.gt(0).to(torch.float32)
        x_mask = x_rgb_mask * x_depth_mask

        x_concat = self.theta_block_11*x_rgb + (1-self.theta_block_11)*x_depth
        
        x = x + x_mask * x_concat
        x_rgb = x_rgb * x_rgb_mask # try comment this line
        x_depth = x_depth * x_depth_mask # try comment this line

        x = self.concat_block_11(x)
        x_rgb = self.rgb_conv_block_11(x_rgb)
        x_depth = self.depth_conv_block_11(x_depth)
    
        # final block conv
        x = self.concat_model_final(x)

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


    def _fix_concat_layer(self, layer):
        weights = layer[0].weight
        kernel_size = tuple(layer[0].kernel_size)
        stride = tuple(layer[0].stride)
        out_features = weights.shape[0]
        new_features = torch.nn.Conv2d(
            out_features, out_features, kernel_size=kernel_size, stride=1, bias=False
        )
        torch.nn.init.xavier_uniform_(new_features.weight)
        return new_features
