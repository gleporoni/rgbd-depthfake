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
from model.maskdepthfake import MaskDepthFake
from torchvision import utils, transforms, io
import torch
from gradcam import GradCAM
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable



log = logging.getLogger(__name__)


def test(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.run.seed)

    # data module declaration
    # data = FaceForensicsPlusPlus(conf)
    # data.setup(stage="test")

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
        "depth_mask_xception",
        "depth_mask_mobilenet",
    ):
        model = MaskDepthFake(conf)
    else:
        raise NotImplementedError

    # trainer
    # trainer: Trainer = hydra.utils.instantiate(conf.run.pl_trainer)

    # Load a pretrained model from a checkpoint
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
    
    # feature_extractor = model.concat_layer

    #find data activation
    # print(feature_extractor.weight.data.shape)
    # for i in range(728):
    #     print(torch.sum(torch.abs(torch.flatten(feature_extractor.weight.data[i][:728]))))
    #     print(torch.sum(torch.abs(torch.flatten(feature_extractor.weight.data[i][728:]))))
    #     print("--------------------------")

    # 0 real - 1 fake
    # iter_dataloader = iter(data.test_dataloader())
    # tmp = next(iter_dataloader)
    # torch.save(tmp, '/workdir/testfeature/real_tensor.pt')
    # while tmp is not None:
    #     inputs = tmp['image']
    #     classes = tmp['label']
    #     print("input shape")
    #     print(inputs.shape)
    #     print(classes)
    #     if(classes[0] == 1):
    #         print("save fake")
    #         torch.save(tmp, '/workdir/testfeature/fake_tensor.pt')
    #         break
    #     tmp = next(iter_dataloader)


    real_tensor = torch.load('/workdir/testfeature/real_tensor.pt')
    fake_tensor = torch.load('/workdir/testfeature/fake_tensor.pt')

    model.eval()

    real_res = model(real_tensor['image'][:, :, :, :]).argmax(dim=1)
    # real_acc = model.test_accuracy(real_res, real_tensor['label'])

    fake_res = model(fake_tensor['image'][:, :, :, :]).argmax(dim=1)
    # fake_acc = model.test_accuracy(fake_res, fake_tensor['label'])

    print("----  real batch ---")
    print(real_res)
    print("----  fake batch ---")
    print(fake_res)


    grad_cam(model, model.rgb_model_final[3], 0, real_tensor['image'][:, :, :, :], '/workdir/testfeature/rgb_feature_real.png')
    grad_cam(model, model.rgb_model_final[3], 1, fake_tensor['image'][:, :, :, :], '/workdir/testfeature/rgb_feature_fake.png')
    grad_cam(model, model.depth_model_final[3], 0, real_tensor['image'][:, :, :, :], '/workdir/testfeature/depth_feature_real.png')
    grad_cam(model, model.depth_model_final[3], 1, fake_tensor['image'][:, :, :, :], '/workdir/testfeature/depth_feature_fake.png')
    
    # real_res[:, 0].backward()

    # cut_model = torch.nn.Sequential(*(list(self.model.model.children())[:24]))
    # gradients = cut_model(real_tensor['image'][:, :3, :, :])



    # save_fusion_grid(real_tensor['image'][:, :3, :, :], cam, '/workdir/testfeature/rgb_feature_real.png')

    # filepath_rgb = Path("/workdir/testfeature/rgb_input_real.png")
    # filepath_depth = Path("/workdir/testfeature/depth_input_real.png")

    # for inputs in [real_tensor['image'], fake_tensor['image']]:
    #     if conf.data.use_depth:
    #         img_grid_rgb = utils.make_grid(inputs[:, :3, :, :]/255, nrow=8, padding=20)

    #         depth = inputs[:, 3, :, :]
    #         max = torch.max(torch.flatten(depth))
    #         img_grid_depth = utils.make_grid(depth[:, None, :, :]/max, nrow=8, padding=20)
    #         utils.save_image(img_grid_depth, filepath_depth)

    #     else: 
    #         img_grid_rgb = utils.make_grid(inputs[:, :, :, :], nrow=8, padding=20)

    #     utils.save_image(img_grid_rgb, filepath_rgb)
    #     filepath_rgb = Path("/workdir/testfeature/rgb_input_fake.png")
    #     filepath_depth = Path("/workdir/testfeature/depth_input_fake.png")


    # if conf.data.use_depth:
    #     img_grid_rgb = utils.make_grid(inputs[:16, :3, :, :]/255, nrow=4, padding=10)


    #     depth = inputs[:, 3, :, :]
    #     max = torch.max(torch.flatten(depth))
    #     img_grid_depth = utils.make_grid(depth[:16, None, :, :]/max, nrow=4, padding=10)
    #     filepath_depth = Path("/workdir/testfeature/depth_input.png")
    #     utils.save_image(img_grid_depth, filepath_depth)

    # else: 
    #      img_grid_rgb = utils.make_grid(inputs[:16, :, :, :], nrow=4, padding=10)

    # utils.save_image(img_grid_rgb, filepath_rgb)

    # if conf.model.model_name == 'depth_double_xception':
    #     x_rgb = model.rgb_model(inputs[:16,:3,:,:])
    #     x_depth = model.depth_model(inputs[:16,3:,:,:])

    #     x = torch.cat((x_rgb, x_depth), dim = 1)
    #     features = model.concat_layer(x)
    # elif conf.model.model_name == 'depth_attention':
    #     features = depth_attention(model, inputs[:16, :, :, :])

    # else:
    #     feature_extractor = torch.nn.Sequential(*(list(model.model.children())[:17]))
    #     features = feature_extractor(inputs[:16, :, :, :])

    # if conf.model.model_name == 'depth_attention':
    #     names = [["features_rgb_1", "features_depth_1", "features_mix_1"],
    #         ["features_rgb_2", "features_depth_2", "features_mix_2"],
    #         ["features_rgb_3", "features_depth_3", "features_mix_3"],
    #         ["features_rgb_4", "features_depth_4", "features_mix_4"]]
    #     for group, subname in zip(features, names):
    #         for feature, name in zip(group, subname):
    #             processed = []
    #             for feature_map in feature:
    #                 feature_map = feature_map.squeeze(0)
    #                 gray_scale = torch.sum(feature_map,0)
    #                 gray_scale = gray_scale / feature_map.shape[0]
    #                 processed.append(gray_scale.data.cpu().numpy())
    #             processed = torch.Tensor(processed)
    #             img_grid_features = utils.make_grid(processed[:16, None, :, :], nrow=4, padding=10)
    #             filepath_features = Path("/workdir/testfeature/"+name+".png")
    #             utils.save_image(img_grid_features, filepath_features) 

    # else:

    #     print(features.shape)
    #     processed = []
    #     for feature_map in features:
    #         feature_map = feature_map.squeeze(0)
    #         gray_scale = torch.sum(feature_map,0)
    #         gray_scale = gray_scale / feature_map.shape[0]
    #         processed.append(gray_scale.data.cpu().numpy())
    #     processed = torch.Tensor(processed)
    #     print(processed.shape)

    #     img_grid_features = utils.make_grid(processed[:16, None, :, :], nrow=4, padding=10)
    #     filepath_features = Path("/workdir/testfeature/features.png")
    #     utils.save_image(img_grid_features, filepath_features)



# def depth_attention(model, x):
#     x_rgb = x[:,:3,:,:]
#     x_depth = x[:,3:,:,:]

#     rgb = model.conv_rgb(x_rgb)
#     depth = model.conv_depth(x_depth)
#     atten_rgb = model.atten_rgb_0(rgb)
#     atten_depth = model.atten_depth_0(depth)
#     m0 = rgb.mul(atten_rgb) + depth.mul(atten_depth)

#     rgb = model.maxpool_rgb(rgb)
#     depth = model.maxpool_detph(depth)
#     m = model.maxpool_mix(m0)

#     # block 1
#     rgb = model.layer1_rgb(rgb)
#     depth = model.layer1_depth(depth)
#     m = model.layer1_mix(m)

#     features_rgb_1 = rgb
#     features_depth_1 = depth
#     features_mix_1 = m

#     atten_rgb = model.atten_rgb_1(rgb)
#     atten_depth = model.atten_depth_1(depth)
#     m1 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

#     # block 2
#     rgb = model.layer2_rgb(rgb)
#     depth = model.layer2_depth(depth)
#     m = model.layer2_mix(m1)

#     features_rgb_2 = rgb
#     features_depth_2 = depth
#     features_mix_2 = m

#     atten_rgb = model.atten_rgb_2(rgb)
#     atten_depth = model.atten_depth_2(depth)
#     m2 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

#     # block 3
#     rgb = model.layer3_rgb(rgb)
#     depth = model.layer3_depth(depth)
#     m = model.layer3_mix(m2)

#     features_rgb_3 = rgb
#     features_depth_3 = depth
#     features_mix_3 = m

#     atten_rgb = model.atten_rgb_3(rgb)
#     atten_depth = model.atten_depth_3(depth)
#     m3 = m + rgb.mul(atten_rgb) + depth.mul(atten_depth)

#     # block 4
#     rgb = model.layer4_rgb(rgb)
#     depth = model.layer4_depth(depth)
#     m = model.layer4_mix(m3)

#     features_rgb_4 = rgb
#     features_depth_4 = depth
#     features_mix_4 = m

#     return [[features_rgb_1, features_depth_1, features_mix_1],
#             [features_rgb_2, features_depth_2, features_mix_2],
#             [features_rgb_3, features_depth_3, features_mix_3],
#             [features_rgb_4, features_depth_4, features_mix_4]]

# def compute_distance(im1, im2):
#     x1 = im1.flatten()
#     x2 = im2.flatten()

#     return np.linalg.norm(x1-x2)
    


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    log.info(OmegaConf.to_yaml(conf))
    test(conf)

def grad_cam(model, target_layer, target_class, input, filepath):
    model.eval()

    # target_layer = model.model.bn4

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    target_layer.register_forward_hook(get_activation('target_layer'))


    output = model(input[:, :, :, :])


    # Target class
    # target_class = real_tensor['label']
    # target_class = 0


    one_hot = torch.zeros((32, output.size()[-1]), dtype=torch.float32)
    one_hot[0][target_class] = 1
    one_hot = Variable(one_hot, requires_grad=True)

    model.zero_grad()
    
    output.backward(gradient=one_hot, retain_graph=True)

    grads = activation['target_layer'].squeeze()

    C, H, W = grads.size(1), grads.size(2), grads.size(3)
    weights = torch.mean(grads, dim=(2, 3))  # shape (32, C)
    weights = weights.view(-1, C, 1, 1)  

    # Apply ReLU to the weights
    weights = weights.clamp(min=0)

    cam = torch.sum(weights * grads, dim=1)

    # Apply ReLU to the activation map
    cam = cam.clamp(min=0)

    # Normalize the activation map
    cam = cam / torch.max(cam)
    cam = cam[:, None, :, :]
    print(cam.shape)

    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    print(cam.shape)

    grid = create_grid(input[:, :3, :, :], cam)

    # filepath_rgb = '/workdir/testfeature/rgb_feature_real.png'

    utils.save_image(grid, filepath)

def heatmap_to_color_image(heatmap):
    heatmap = np.squeeze(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = plt.cm.jet(heatmap)
    heatmap = np.delete(heatmap, 3, 2)
    heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
    return heatmap

import torchvision.transforms.functional as TF
from PIL import Image


def create_grid(images, heatmaps):
    # Convert the heatmaps to color images
    color_heatmaps = [heatmap_to_color_image(heatmap) for heatmap in heatmaps]

    # for a, b in zip(images, color_heatmaps):
    #     print(a.shape)
    #     print(b.shape)

    # Combine the images and color heatmaps
    
    combined_images = [TF.to_tensor(Image.blend(TF.to_pil_image(image), TF.to_pil_image(color_heatmap), 0.5)) for image, color_heatmap in zip(images, color_heatmaps)]

    # Create the grid of combined images
    grid = utils.make_grid(combined_images, nrow=8 , normalize=False, scale_each=False, padding=20)

    return grid


if __name__ == "__main__":
    main()
