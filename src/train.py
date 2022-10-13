import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from data.data_loader import FaceForensicsPlusPlus
from model.rgb import RGB


def train(conf: omegaconf.DictConfig, model_name: str) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = FaceForensicsPlusPlus(conf)

    # main module declaration
    pl_module = RGB(conf, model_name)

    # callbacks declaration
    callbacks_store = []

    # if conf.train.early_stopping_callback is not None:
    #     early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
    #     callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer, callbacks=callbacks_store)

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    for model_name in conf.model.backbone:
        train(conf, model_name)


if __name__ == "__main__":
    main()
