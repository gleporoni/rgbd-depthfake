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


log = logging.getLogger(__name__)


def test(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.run.seed)

    # data module declaration
    # data = FaceForensicsPlusPlus(conf)
    # data.setup(stage="fit")

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
    elif conf.model.model_name in (
        "depth_attention",
    ):
        model = AttentionDepthFake(conf)
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
    model = model.load_from_checkpoint(checkpoint_path="/workdir/weights/depth_double_xception_exit/epoch=0-step=630.ckpt" )
    
    feature_extractor = model.concat_layer

    #find data activation
    # print(feature_extractor.weight.data.shape)
    # for i in range(728):
    #     print(torch.sum(torch.abs(torch.flatten(feature_extractor.weight.data[i][:728]))))
    #     print(torch.sum(torch.abs(torch.flatten(feature_extractor.weight.data[i][728:]))))
    #     print("--------------------------")



    tmp = next(iter(data.train_dataloader()))
    inputs = tmp['image']
    classes = tmp['label']
    print("input shape")
    print(inputs.shape)

    filepath_rgb = Path("/workdir/testfeature/rgb_input.png")


    if conf.data.use_depth:
        img_grid_rgb = utils.make_grid(inputs[:16, :3, :, :]/255, nrow=4, padding=10)


        depth = inputs[:, 3, :, :]
        max = torch.max(torch.flatten(depth))
        img_grid_depth = utils.make_grid(depth[:16, None, :, :]/max, nrow=4, padding=10)
        filepath_depth = Path("/workdir/testfeature/depth_input.png")
        utils.save_image(img_grid_depth, filepath_depth)

    else: 
         img_grid_rgb = utils.make_grid(inputs[:16, :, :, :], nrow=4, padding=10)

    utils.save_image(img_grid_rgb, filepath_rgb)

    if conf.model.model_name == 'depth_double_xception':
        x_rgb = model.rgb_model(inputs[:16,:3,:,:])
        x_depth = model.depth_model(inputs[:16,3:,:,:])

        x = torch.cat((x_rgb, x_depth), dim = 1)
        features = model.concat_layer(x)
    elif conf.model.model_name == 'depth_attention':
        features = depth_attention(model, inputs[:16, :, :, :])

    else:
        feature_extractor = torch.nn.Sequential(*(list(model.model.children())[:17]))
        features = feature_extractor(inputs[:16, :, :, :])

    if conf.model.model_name == 'depth_attention':
        names = [["features_rgb_1", "features_depth_1", "features_mix_1"],
            ["features_rgb_2", "features_depth_2", "features_mix_2"],
            ["features_rgb_3", "features_depth_3", "features_mix_3"],
            ["features_rgb_4", "features_depth_4", "features_mix_4"]]
        for group, subname in zip(features, names):
            for feature, name in zip(group, subname):
                processed = []
                for feature_map in feature:
                    feature_map = feature_map.squeeze(0)
                    gray_scale = torch.sum(feature_map,0)
                    gray_scale = gray_scale / feature_map.shape[0]
                    processed.append(gray_scale.data.cpu().numpy())
                processed = torch.Tensor(processed)
                img_grid_features = utils.make_grid(processed[:16, None, :, :], nrow=4, padding=10)
                filepath_features = Path("/workdir/testfeature/"+name+".png")
                utils.save_image(img_grid_features, filepath_features) 

    else:

        print(features.shape)
        processed = []
        for feature_map in features:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        processed = torch.Tensor(processed)
        print(processed.shape)

        img_grid_features = utils.make_grid(processed[:16, None, :, :], nrow=4, padding=10)
        filepath_features = Path("/workdir/testfeature/features.png")
        utils.save_image(img_grid_features, filepath_features)



def depth_attention(model, x):
    x_rgb = x[:,:3,:,:]
    x_depth = x[:,3:,:,:]

    rgb = model.conv_rgb(x_rgb)
    depth = model.conv_depth(x_depth)
    atten_rgb = model.atten_rgb_0(rgb)
    atten_depth = model.atten_depth_0(depth)
    m0 = rgb.mul(atten_rgb) + depth.mul(atten_depth)

    rgb = model.maxpool_rgb(rgb)
    depth = model.maxpool_detph(depth)
    m = model.maxpool_mix(m0)

    # block 1
    rgb = model.layer1_rgb(rgb)
    depth = model.layer1_depth(depth)
    m = model.layer1_mix(m)

    features_rgb_1 = rgb
    features_depth_1 = depth
    features_mix_1 = m

    atten_rgb = model.atten_rgb_1(rgb)
    atten_depth = model.atten_depth_1(depth)
    m1 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

    # block 2
    rgb = model.layer2_rgb(rgb)
    depth = model.layer2_depth(depth)
    m = model.layer2_mix(m1)

    features_rgb_2 = rgb
    features_depth_2 = depth
    features_mix_2 = m

    atten_rgb = model.atten_rgb_2(rgb)
    atten_depth = model.atten_depth_2(depth)
    m2 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

    # block 3
    rgb = model.layer3_rgb(rgb)
    depth = model.layer3_depth(depth)
    m = model.layer3_mix(m2)

    features_rgb_3 = rgb
    features_depth_3 = depth
    features_mix_3 = m

    atten_rgb = model.atten_rgb_3(rgb)
    atten_depth = model.atten_depth_3(depth)
    m3 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

    # block 4
    rgb = model.layer4_rgb(rgb)
    depth = model.layer4_depth(depth)
    m = model.layer4_mix(m3)

    features_rgb_4 = rgb
    features_depth_4 = depth
    features_mix_4 = m

    return [[features_rgb_1, features_depth_1, features_mix_1],
            [features_rgb_2, features_depth_2, features_mix_2],
            [features_rgb_3, features_depth_3, features_mix_3],
            [features_rgb_4, features_depth_4, features_mix_4]]

def compute_distance(im1, im2):
    x1 = im1.flatten()
    x2 = im2.flatten()

    return np.linalg.norm(x1-x2)
    


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    log.info(OmegaConf.to_yaml(conf))
    test(conf)


if __name__ == "__main__":
    main()


