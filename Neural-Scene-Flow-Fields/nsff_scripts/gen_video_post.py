import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
import os
import imageio
import numpy as np

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    
    scenes = ["DynamicFace-2", "Truck-2","Umbrella", "Balloon1-2", "Balloon2-2", "playground", "Jumping", "Skating-2"]
    root_dir = args.root_dir
    post_dir = root_dir + "_processed"
    for scene in scenes:


        idx = 0
        grids = None 
        while(os.path.exists(os.path.join(root_dir, scene, f"{idx}.png"))):
            raw = read_image(os.path.join(root_dir, scene, f"{idx}.png"), torchvision.io.ImageReadMode.RGB)
            post = read_image(os.path.join(post_dir, scene, f"{idx}.png"), torchvision.io.ImageReadMode.RGB)
            grid = make_grid([raw, post])
            if grids is None:
                grids = grid.permute(1, 2, 0)[None, ...]
            else:
                grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
            idx += 1
        torchvision.io.write_video(os.path.join(post_dir, f"{scene}.mp4"), grids, fps=1)
    #writer.close()
