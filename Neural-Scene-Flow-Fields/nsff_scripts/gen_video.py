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
    
    # first pass, get the PCA to project feature down to 3 channel
    idx = 0 
    dinos = None
    while(os.path.exists(os.path.join(args.root_dir, f"{idx}_dino.pt"))):
        dino = torch.load(os.path.join(args.root_dir, f"{idx}_dino.pt"))

        if dinos is None:
            dinos = dino[None, ...]
        else:
            dinos = torch.cat([dinos, dino[None, ...]], dim=0)
        idx += 1
    if dinos is not None:
        pca = PCA(n_components=3).fit(dinos.view(-1, dinos.shape[-1])[::100])
        old_shape = dinos.shape
        dinos = torch.from_numpy(pca.transform(dinos.view(-1, dinos.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], 3)
    
        for comp_idx in range(3):
            comp = dinos[..., comp_idx]
            comp_min = torch.min(comp)
            comp_max = torch.max(comp)
            comp = (comp - comp_min) / (comp_max - comp_min)
            dinos[..., comp_idx] = comp
   
    #writer = imageio.get_writer(os.path.join(args.root_dir, "output.mp4"), fps=1) 
    idx = 0
    grids = None 
    while(os.path.exists(os.path.join(args.root_dir, f"{idx}_rgb.png"))):
        rgb = read_image(os.path.join(args.root_dir, f"{idx}_rgb.png"), torchvision.io.ImageReadMode.RGB)
        blend = read_image(os.path.join(args.root_dir, f"{idx}_blend.png"), torchvision.io.ImageReadMode.RGB)
        depth = read_image(os.path.join(args.root_dir, f"{idx}_depth.png"), torchvision.io.ImageReadMode.RGB)
        if os.path.exists(os.path.join(args.root_dir, f"{idx}_sal.png")):
            sal = read_image(os.path.join(args.root_dir, f"{idx}_sal.png"), torchvision.io.ImageReadMode.RGB)
        else:
            sal = torch.zeros_like(depth)
        if dinos is not None: 
            dino = dinos[idx].permute(2, 0, 1)*255.
        else:
            dino = torch.zeros_like(rgb)
        grid = make_grid([rgb, depth, blend, dino, sal], nrow=2)
        if grids is None:
            grids = grid.permute(1, 2, 0)[None, ...]
        else:
            grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
        idx += 1
    torchvision.io.write_video(os.path.join(args.root_dir, "output.mp4"), grids, fps=1)
    #writer.close()
