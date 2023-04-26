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


def train(conf: omegaconf.DictConfig) -> None:

    os.environ["CUDA_VISIBLE_DEVICES"] = conf.run.cuda_device

    # reproducibility
    pl.seed_everything(conf.run.seed)


    # loggers
    csv_logger = CSVLogger(
        "logs",
        name=f"{conf.model.model_name}_{conf.data.compression_level[0]}_{datetime.now().strftime('%d-%m-%Y %H-%M-%S.%f')}",
    )
    loggers = [csv_logger]

    # # data module declaration
    data = FaceForensicsPlusPlus(conf)
    data.setup(stage="fit")
    log.info(f"Train data: {len(data.train_data)}")
    log.info(f"Val data: {len(data.val_data)}")

    # main module declaration
    if conf.model.model_name in (
        "rgb_efficientnet",
        "rgb_mobilenet",
        "rgb_resnet",
        "rgb_shufflenet",
        "rgb_vit",
        "rgb_xception",
    ):
        model = RGB(conf)
    elif conf.model.model_name in (
        "depth_efficientnet",
        "depth_mobilenet",
        "depth_resnet",
        "depth_shufflenet",
        "depth_vit",
        "depth_xception",
    ):
        model = DepthFake(conf)
    elif conf.model.model_name in (
        "depth_double_xception",
        "depth_double_mobilenet",
    ):
        model = DoubleDepthFake(conf)
    elif conf.model.model_name in (
        "depth_attention",
    ):
        model = AttentionDepthFake(conf)    
    elif conf.model.model_name in (
        "depth_rocketnet",
    ):
        model = DepthFakeRocket(conf)
    elif conf.model.model_name in (
        "depth_mask_xception",
        "depth_mask_mobilenet",
    ):
        model = MaskDepthFake(conf)
    else:
        raise NotImplementedError


    # log gradients and model topology
    if (
        "fast_dev_run" not in conf.run.pl_trainer
        and "overfit_batches" not in conf.run.pl_trainer
    ):  # i.e. if not developing
        wandb_logger = WandbLogger(
            project=conf.project,
            name=f"{conf.model.model_name}_{conf.data.compression_level[0]}_{datetime.now().strftime('%d-%m-%Y %H-%M-%S.%f')}",
        )
        loggers.append(wandb_logger)
        wandb_logger.watch(model)

    # callbacks declaration
    callbacks_store = []

    if conf.run.model_checkpoint_callback_loss is not None:
        model_checkpoint_callback_loss: ModelCheckpoint = hydra.utils.instantiate(
            conf.run.model_checkpoint_callback_loss
        )
        callbacks_store.append(model_checkpoint_callback_loss)

    if conf.run.model_checkpoint_callback_acc is not None:
        model_checkpoint_callback_acc: ModelCheckpoint = hydra.utils.instantiate(
            conf.run.model_checkpoint_callback_acc
        )

        callbacks_store.append(model_checkpoint_callback_acc)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.run.pl_trainer, callbacks=callbacks_store, logger=loggers
    )

    # load pretrained model 
    if conf.run.experiment.pretrain:
        base_path = Path(Path(__file__).parent, "../")
        checkpoint_path = Path(
            base_path,
            conf.run.experiment.checkpoint_file,
        )
        try:
            model = model.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=False)
        except:
            checkpoint = torch.load(checkpoint_path)
            new_weights = model.state_dict()
            old_weights = list(checkpoint['state_dict'].items())
            i=0
            for k, _ in new_weights.items():
                new_weights[k] = old_weights[i][1]
                i += 1
            model.load_state_dict(new_weights)


    # module fit
    trainer.fit(model, datamodule=data)

    # module test
    data.setup(stage="test")
    log.info(f"Test data: {len(data.test_data)}")
    trainer.test(model, datamodule=data)

    trainer.save_checkpoint(filepath=conf.run.path_last, weights_only = False)


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    # log gradients and model topology
    if (
        "fast_dev_run" not in conf.run.pl_trainer
        and "overfit_batches" not in conf.run.pl_trainer
    ):  # i.e. if not developing
        wandb.finish(quiet=True)

    log.info(OmegaConf.to_yaml(conf))
    train(conf)


if __name__ == "__main__":
    main()
