import omegaconf
from omegaconf import OmegaConf
import hydra
import logging

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pathlib import Path
from data.data_loader import FaceForensicsPlusPlus
from model.rgb import RGB
from model.depthfake import DepthFake
from model.doubledepthfake import DoubleDepthFake
from model.attentiondepthfake import AttentionDepthFake
from torchvision import utils, transforms, io
import torch
import matplotlib.pyplot as plt
import numpy as np



def start(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.run.seed)


    data = FaceForensicsPlusPlus(conf)
    data.setup(stage="fit")
    

    model1 = DoubleDepthFake(conf)

    model1 = model1.load_from_checkpoint(checkpoint_path="/workdir/experiments/depth_double_xception/2023-03-07/12-47-44/experiments/depth_double_xception/epoch=28-step=18270.ckpt" )
    
    model1.model = torch.nn.Sequential(*(list(model1.model.children())[:16])) 


    i = 0

    for elem in iter(data.train_dataloader()):
        inputs = elem['image']
        classes = elem['label']

        out = model1(inputs)

        torch.save(out, "/media/svmset/train/train_outs_"+str(i)+".pt")
        torch.save(classes, "/media/svmset/train/label_train_outs_"+str(i)+".pt")

        i+=1


    i = 0

    for elem in iter(data.val_dataloader()):
        inputs = elem['image']
        classes = elem['label']

        out = model1(inputs)

        torch.save(out, "/media/svmset/val/val_outs_"+str(i)+".pt")
        torch.save(classes, "/media/svmset/val/label_val_outs_"+str(i)+".pt")

        i += 1


    data = FaceForensicsPlusPlus(conf)
    data.setup(stage="test")

    i = 0

    for elem in iter(data.test_dataloader()):
        inputs = elem['image']
        classes = elem['label']

        out = model1(inputs)

        torch.save(out, "/media/svmset/test/test_outs_"+str(i)+".pt")
        torch.save(classes, "/media/svmset/test/label_test_outs_"+str(i)+".pt")

        i += 1

@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    start(conf)


if __name__ == "__main__":
    main()