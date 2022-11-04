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
    
    wfeat_id = 0
    while os.path.exists(os.path.join(args.root_dir, f"cluster_dino_{wfeat_id}_0")):
        wsal_id = 0
        while os.path.exists(os.path.join(args.root_dir, f"cluster_dino_{wfeat_id}_{wsal_id}")):
            imgss = None
            for scene in ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "Umbrella"]:
                basedir = os.path.join(args.root_dir, f"cluster_dino_{wfeat_id}_{wsal_id}", scene, "train")
                idx = 0
                imgs = []

                while os.path.exists(os.path.join(basedir, f"{idx}.png")):
                    final = read_image(os.path.join(basedir, f"{idx}.png"), torchvision.io.ImageReadMode.RGB)
                    raw = read_image(os.path.join(basedir, f"{idx}_raw.png"), torchvision.io.ImageReadMode.RGB)
                    imgs.append(torch.cat([final, raw], dim=1))    
                    #grid = make_grid([raw, final], nrow=4)
                    #if grids is None:
                    #    grids = grid.permute(1, 2, 0)[None, ...]
                    #else:
                    #    grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
                    idx += 1
                if imgss is None:
                    imgss = torch.stack(imgs, dim=0)
                else:
                    imgss = torch.cat([imgss, torch.stack(imgs, dim=0)], dim=-1)
            if imgss.shape[-1] %2 == 1:
                imgss = imgss[..., :-1]
            torchvision.io.write_video(os.path.join(args.root_dir, f"cluster_dino_{wfeat_id}_{wsal_id}", f"{wfeat_id}_{wsal_id}_output.mp4"), imgss.permute(0, 2, 3, 1), fps=1)
            wsal_id += 1
        wfeat_id += 1 
    