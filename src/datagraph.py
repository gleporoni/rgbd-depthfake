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



    # main module declaration
    # if conf.model.model_name in (
    #     "rgb_efficientnet",
    #     "rgb_mobilenet",
    #     "rgb_resnet",
    #     "rgb_shufflenet",
    #     "rgb_vit",
    #     "rgb_xception",
    # ):
    #     model = RGB(conf)
    # elif conf.model.model_name in (
    #     "depth_efficientnet",
    #     "depth_mobilenet",
    #     "depth_resnet",
    #     "depth_shufflenet",
    #     "depth_vit",
    #     "depth_xception",
    # ):
    #     model = DepthFake(conf)
    # elif conf.model.model_name in (
    #     "depth_double_xception",
    # ):
    #     model = DoubleDepthFake(conf)
    # elif conf.model.model_name in (
    #     "depth_attention",
    # ):
    #     model = AttentionDepthFake(conf)
    # else:
    #     raise NotImplementedError

    

    conf.model.model_name = 'rgb_xception'
    conf.model.backbone = 'xception'
    conf.data.use_hha = False

    model1 = RGB(conf)

    conf.model.model_name = 'depth_xception'
    conf.model.backbone = 'xception'
    conf.data.use_hha = False


    model2 = DepthFake(conf)

    model1 = model1.load_from_checkpoint(checkpoint_path="/workdir/weights/rgb_xception/epoch=0-step=637.ckpt" )
    model2 = model2.load_from_checkpoint(checkpoint_path="/workdir/weights/depth_xception/epoch=0-step=630.ckpt")

    # for layer in model1.model.children():
    #     try:
    #         print(torch.mean(torch.flatten(layer.weight)))
    #     except:
    #         print(layer)

    w1 = []
    w2 = []
    w3 = []

    i = 0

    for name, param in model1.named_parameters():
        # if i == 39:
        #     break
        w1.append(torch.mean(torch.flatten(param)).detach().numpy())
        i +=1

    print("------------")


    i = 0

    # for name, param in model2.named_parameters():
    #     if 'rgb_model' in name:
    #         w2.append(torch.mean(torch.flatten(param)).detach().numpy())
    #     elif 'depth_model' in name:
    #         i+= 1
    #         if i == 2:
    #             continue
    #         w3.append(torch.mean(torch.flatten(param)).detach().numpy())

        # print(name, param.size())
    


    i = 0
    for name, param in model2.named_parameters():
        i+=1
        if i == 2:
            continue
        w2.append(torch.mean(torch.flatten(param)).detach().numpy())
        

    print(len(w1))
    print(len(w2))
    print(len(w3))

    x = range(0, len(w1))

    print(w1)
    print("---")
    
    print(w2)
    print("---")
    
    print(w3)
    print("---")
    

  
    # line 1 points
    # plotting the line 1 points 
    # plt.plot(x, w1, label = "rgb weights")
    
    # plotting the line 2 points 
    plt.plot(x, torch.abs(torch.Tensor(np.array(w1)-np.array(w2))), label = "RGB-Depth")

    # plt.plot(x, w3, label = "doubledepth depth weights")
    
    # naming the x axis
    plt.xlabel('layers')
    # naming the y axis
    plt.ylabel('weights average')
    # giving a title to my graph
    plt.title('weights average')
    
    # show a legend on the plot
    plt.legend()
    
    # function to show the plot
    plt.savefig('/workdir/plot4.png')

@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(conf: omegaconf.DictConfig):
    start(conf)


if __name__ == "__main__":
    main()