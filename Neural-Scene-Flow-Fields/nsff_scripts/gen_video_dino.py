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
    

    scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "Umbrella"]
    splits = ["training", "nv_spatial", "nv_static"]
    
    for scene in scenes:
        
        for split in splits:
            dino_dir = os.path.join(args.root_dir, scene, split)
            for level in ["", "_1", "_2"]:
                feats = torch.load(os.path.join(dino_dir, f"feats{level}.pt"))
                sals = torch.load(os.path.join(dino_dir, f"sals{level}.pt"))
                counter = torch.load(os.path.join(dino_dir, f"counter{level}.pt"))
                #assert False, [feats.shape, sals.shape, counter.shape]
            
                feats /= counter + 1e-16
                sals /= counter + 1e-16
                pca = PCA(n_components=3).fit(feats.view(-1, feats.shape[-1])[::100])
                old_shape = feats.shape
                feats = torch.from_numpy(pca.transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], 3)
        
                for comp_idx in range(3):
                    comp = feats[..., comp_idx]
                    comp_min = torch.min(comp)
                    comp_max = torch.max(comp)
                    comp = (comp - comp_min) / (comp_max - comp_min)
                    feats[..., comp_idx] = comp
                #assert False, [feats.shape, sals.shape]
                #assert False, torchvision.transforms.functional.to_pil_image(feats[0].permute(2, 0, 1)).save("test.png")
                grids = None
                for idx in range(len(feats)):
                    feat = feats[idx]
                    sal = sals[idx]
                    grid = make_grid([feat.permute(2, 0, 1)*255., sal.permute(2, 0, 1).repeat(3, 1, 1)*255.])
                    #assert False, [feat.shape, sal.shape, grid.shape]
                    #assert False, grid.shape
                    if grids is None:
                        grids = grid.permute(1, 2, 0)[None, ...]
                    else:
                        grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
        
                torchvision.io.write_video(os.path.join(dino_dir, f"{level}_.mp4"), 
                grids, fps=1)
        