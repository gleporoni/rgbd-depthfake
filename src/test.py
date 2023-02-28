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



log = logging.getLogger(__name__)


def test(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.run.seed)

    # data module declaration
    data = FaceForensicsPlusPlus(conf)
    data.setup(stage="test")

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
    ):
        model = DoubleDepthFake(conf)
    else:
        raise NotImplementedError

    # trainer
    trainer: Trainer = hydra.utils.instantiate(conf.run.pl_trainer)

    # Load a pretrained model from a checkpoint
    base_path = Path(Path(__file__).parent, "../")
    checkpoint_path = Path(
        base_path,
        conf.run.experiment.checkpoint_file,
    )
    model = model.load_from_checkpoint(checkpoint_path="/workdir/weights/test.ckpt" )


    # module test
    trainer.test(model, datamodule=data)


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    log.info(OmegaConf.to_yaml(conf))
    test(conf)


if __name__ == "__main__":
    main()
