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






class DepthFakeRocket(pl.LightningModule):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        self.num_classes = self.conf.data.num_classes
        in_features = 1
        if self.conf.data.depth_type  == 'hha':
            in_features = 3

        if self.conf.model.backbone == "xception":
            self.model = timm.create_model(
                "xception", pretrained=True, num_classes=self.num_classes
            )

            # self.rgb_model = RGB(conf)
            tmp_model = list(self.model.children())

            # net before block 1
            self.rgb_model_init = copy.deepcopy(torch.nn.Sequential(*(tmp_model[:6])))
            self.depth_model_init = copy.deepcopy(torch.nn.Sequential(*(tmp_model[:6])))

            if in_features == 1:
            #remove rgb input and take only depth input in the first layer
                weights = self.depth_model_init[0].weight
                kernel_size = tuple(self.depth_model_init[0].kernel_size)
                stride = tuple(self.depth_model_init[0].stride)
                out_features = weights.shape[0]
                new_features = torch.nn.Conv2d(
                    in_features, out_features, kernel_size=kernel_size, stride=stride
                )
                torch.nn.init.xavier_uniform_(new_features.weight)
                self.depth_model_init[0] = new_features


            # block 1
            tmp = list(tmp_model[6].children())
            self.rgb_skip_block_1 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.rgb_conv_block_1 = copy.deepcopy(tmp[2])

            self.depth_skip_block_1 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.depth_conv_block_1 = copy.deepcopy(tmp[2])

            self.concat_block_1 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.concat_block_1[0] = self._fix_concat_layer(self.concat_block_1)
                            
            # block 2
            tmp = list(tmp_model[7].children())
            self.rgb_skip_block_2 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.rgb_conv_block_2 = copy.deepcopy(tmp[2])

            self.depth_skip_block_2 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.depth_conv_block_2 = copy.deepcopy(tmp[2])

            self.concat_block_2 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.concat_block_2[0] = self._fix_concat_layer(self.concat_block_2)

            
            # block 3
            tmp = list(tmp_model[8].children())
            self.rgb_skip_block_3 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.rgb_conv_block_3 = copy.deepcopy(tmp[2])

            self.depth_skip_block_3 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.depth_conv_block_3 = copy.deepcopy(tmp[2])

            self.concat_block_3 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.concat_block_3[0] = self._fix_concat_layer(self.concat_block_3)

            # block 4
            self.rgb_conv_block_4 = copy.deepcopy(list(tmp_model[9].children())[0])
            self.depth_conv_block_4 = copy.deepcopy(list(tmp_model[9].children())[0])

            tmp_conv = torch.nn.Conv2d(728, 728, kernel_size=3, stride=1, padding=1 bias=False)
            tmp_batchnorm = torch.nn.BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            torch.nn.init.xavier_uniform_(tmp_conv.weight)

            self.concat_block_4 = torch.nn.Sequential(*([tmp_conv, tmp_batchnorm]))

            # block 5
            self.rgb_conv_block_5 = copy.deepcopy(list(tmp_model[10].children())[0])
            self.depth_conv_block_5 = copy.deepcopy(list(tmp_model[10].children())[0])

            self.concat_block_5 = copy.deepcopy(torch.nn.Sequential(*([tmp_conv, tmp_batchnorm])))

            # block 6
            self.rgb_conv_block_6 = copy.deepcopy(list(tmp_model[11].children())[0])
            self.depth_conv_block_6 = copy.deepcopy(list(tmp_model[11].children())[0])

            self.concat_block_6 = copy.deepcopy(torch.nn.Sequential(*([tmp_conv, tmp_batchnorm])))

            # block 7
            self.rgb_conv_block_7 = copy.deepcopy(list(tmp_model[12].children())[0])
            self.depth_conv_block_7 = copy.deepcopy(list(tmp_model[12].children())[0])

            self.concat_block_7 = copy.deepcopy(torch.nn.Sequential(*([tmp_conv, tmp_batchnorm])))

            # block 8
            self.rgb_conv_block_8 = copy.deepcopy(list(tmp_model[13].children())[0])
            self.depth_conv_block_8 = copy.deepcopy(list(tmp_model[13].children())[0])

            self.concat_block_8 = copy.deepcopy(torch.nn.Sequential(*([tmp_conv, tmp_batchnorm])))

            # block 9
            self.rgb_conv_block_9 = copy.deepcopy(list(tmp_model[14].children())[0])
            self.depth_conv_block_9 = copy.deepcopy(list(tmp_model[14].children())[0])

            self.concat_block_9 = copy.deepcopy(torch.nn.Sequential(*([tmp_conv, tmp_batchnorm])))

            # block 10
            self.rgb_conv_block_10 = copy.deepcopy(list(tmp_model[15].children())[0])
            self.depth_conv_block_10 = copy.deepcopy(list(tmp_model[15].children())[0])

            self.concat_block_10 = copy.deepcopy(torch.nn.Sequential(*([tmp_conv, tmp_batchnorm])))

            # block 11
            self.rgb_conv_block_11 = copy.deepcopy(list(tmp_model[16].children())[0])
            self.depth_conv_block_11 = copy.deepcopy(list(tmp_model[16].children())[0])

            self.concat_block_11 = copy.deepcopy(torch.nn.Sequential(*([tmp_conv, tmp_batchnorm])))

            # block 12
            tmp = list(tmp_model[17].children())
            self.rgb_skip_block_12 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.rgb_conv_block_12 = copy.deepcopy(tmp[2])

            self.depth_skip_block_12 =copy.deepcopy( torch.nn.Sequential(*(tmp[:2])))
            self.depth_conv_block_12 = copy.deepcopy(tmp[2])

            self.concat_block_12 = copy.deepcopy(torch.nn.Sequential(*(tmp[:2])))
            self.concat_block_12[0] = self._fix_concat_layer(self.concat_block_12)
            
            # net after block 12
            self.rgb_model_final = copy.deepcopy(torch.nn.Sequential(*(tmp_model[18:25])))
            self.depth_model_final = copy.deepcopy(torch.nn.Sequential(*(tmp_model[18:25])))

            self.fc_layer = torch.nn.Linear(in_features = 4096, out_features = 2, bias = True)
            torch.nn.init.xavier_uniform_(self.fc_layer.weight)

            self.model = None


        else:
            raise NotImplementedError

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters(conf)

    def forward(self, x) -> dict:
        x_rgb = self.rgb_model_init(x[:,:3,:,:])
        x_depth = self.depth_model_init(x[:,3:,:,:])
        
        # block 1
        x_rgb_skip = self.rgb_skip_block_1(x_rgb)
        x_depth_skip = self.depth_skip_block_1(x_depth)

        x_skip = self.concat_block_1(x_rgb_skip+x_depth_skip)

        x_rgb = self.rgb_conv_block_1(x_rgb) + x_skip
        x_depth = self.depth_conv_block_1(x_depth) + x_skip

        # block 2
        x_rgb_skip = self.rgb_skip_block_2(x_rgb)
        x_depth_skip = self.depth_skip_block_2(x_depth)

        x_skip = self.concat_block_2(x_rgb_skip+x_depth_skip)

        x_rgb = self.rgb_conv_block_2(x_rgb) + x_skip
        x_depth = self.depth_conv_block_2(x_depth) + x_skip

        # block 3
        x_rgb_skip = self.rgb_skip_block_3(x_rgb)
        x_depth_skip = self.depth_skip_block_3(x_depth)

        x_skip = self.concat_block_3(x_rgb_skip+x_depth_skip)

        x_rgb = self.rgb_conv_block_3(x_rgb) + x_skip
        x_depth = self.depth_conv_block_3(x_depth) + x_skip

        # block 4
        x_skip = self.concat_block_4(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_4(x_rgb) + x_skip
        x_depth = self.depth_conv_block_4(x_depth) + x_skip

        # block 5
        x_skip = self.concat_block_5(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_5(x_rgb) + x_skip
        x_depth = self.depth_conv_block_5(x_depth) + x_skip

        # block 6
        x_skip = self.concat_block_6(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_6(x_rgb) + x_skip
        x_depth = self.depth_conv_block_6(x_depth) + x_skip

        # block 7
        x_skip = self.concat_block_7(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_7(x_rgb) + x_skip
        x_depth = self.depth_conv_block_7(x_depth) + x_skip

        # block 8
        x_skip = self.concat_block_8(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_8(x_rgb) + x_skip
        x_depth = self.depth_conv_block_8(x_depth) + x_skip

        # block 9
        x_skip = self.concat_block_9(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_9(x_rgb) + x_skip
        x_depth = self.depth_conv_block_9(x_depth) + x_skip

        # block 10
        x_skip = self.concat_block_10(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_10(x_rgb) + x_skip
        x_depth = self.depth_conv_block_10(x_depth) + x_skip

        # block 11
        x_skip = self.concat_block_11(x_rgb+x_depth)
        x_rgb = self.rgb_conv_block_11(x_rgb) + x_skip
        x_depth = self.depth_conv_block_11(x_depth) + x_skip

        # block 12
        x_rgb_skip = self.rgb_skip_block_12(x_rgb)
        x_depth_skip = self.depth_skip_block_12(x_depth)

        x_skip = self.concat_block_12(x_rgb_skip+x_depth_skip)

        x_rgb = self.rgb_conv_block_12(x_rgb) + x_skip
        x_depth = self.depth_conv_block_12(x_depth) + x_skip

        # net final part
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


    def _fix_concat_layer(self, layer):
        weights = layer[0].weight
        kernel_size = tuple(layer[0].kernel_size)
        stride = tuple(layer[0].stride)
        out_features = weights.shape[0]
        new_features = torch.nn.Conv2d(
            out_features, out_features, kernel_size=3, stride=1, padding=1 bias=False
        )
        torch.nn.init.xavier_uniform_(new_features.weight)
        return new_features
