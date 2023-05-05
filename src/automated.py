import sys
import omegaconf
from omegaconf import OmegaConf
import hydra
import logging
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb

from pathlib import Path
from data.data_loader import FaceForensicsPlusPlus
from model.rgb import RGB
from model.depthfake import DepthFake
from model.doubledepthfake import DoubleDepthFake
from model.attentiondepthfake import AttentionDepthFake
from model.depthfakerocket import DepthFakeRocket
from model.maskdepthfake import MaskDepthFake
from torchvision.utils import save_image
import torch
import os

log = logging.getLogger(__name__)

@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig) -> None:
    log.info(OmegaConf.to_yaml(conf))
    print(conf.run.cuda_device)
    print(conf.model)
    print(conf.data.input_type)
    print(conf.data.attacks)

if __name__ == "__main__":
    main()
