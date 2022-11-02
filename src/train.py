import omegaconf
from omegaconf import OmegaConf
import hydra
import logging

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from data.data_loader import FaceForensicsPlusPlus
from model.rgb import RGB


log = logging.getLogger(__name__)

def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.run.seed)

    # logger
    wandb_logger = (
        WandbLogger(project=conf.project)
        if "fast_dev_run" not in conf.run.pl_trainer
        and "overfit_batches" not in conf.run.pl_trainer  # i.e. if not developing
        else True
    )

    # data module declaration
    data = FaceForensicsPlusPlus(conf)
    data.setup(stage="fit")

    # main module declaration
    model = RGB(conf)
    # log gradients and model topology
    if wandb_logger is not None and type(wandb_logger) is not bool:
        wandb_logger.watch(model)


    # callbacks declaration
    callbacks_store = []

    if conf.run.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(
            conf.run.model_checkpoint_callback
        )
        callbacks_store.append(model_checkpoint_callback)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.run.pl_trainer, callbacks=callbacks_store, logger=wandb_logger
    )

    # module fit
    trainer.fit(model, datamodule=data)

    # module test
    data.setup(stage="test")
    trainer.test(model, datamodule=data)


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    log.info(OmegaConf.to_yaml(conf))
    train(conf)


if __name__ == "__main__":
    main()
