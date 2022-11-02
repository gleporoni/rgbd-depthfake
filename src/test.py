import omegaconf
from omegaconf import OmegaConf
import hydra
import logging

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pathlib import Path
from data.data_loader import FaceForensicsPlusPlus
from model.rgb import RGB


log = logging.getLogger(__name__)

def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.run.seed)

    # data module declaration
    data = FaceForensicsPlusPlus(conf)

    # main module declaration
    model = RGB(conf)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.run.pl_trainer
    )

    # Load a pretrained model from a checkpoint
    base_path = Path(Path(__file__).parent, "../")
    checkpoint_path = Path(
        base_path,
        conf.run.experiment.checkpoint_file,
    )
    model.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )

    # module test
    trainer.test(model, datamodule=data)

    


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    log.info(OmegaConf.to_yaml(conf))
    train(conf)


if __name__ == "__main__":
    main()
