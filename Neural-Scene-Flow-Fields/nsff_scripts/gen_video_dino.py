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
            '''
            for level in ["", "_1", "_2"]:
                time_idx = 0
                featss = None
                while os.path.exists(os.path.join(dino_dir, f"{time_idx}_feats{level}.pt")):
                    feats = torch.load(os.path.join(dino_dir, f"{time_idx}_feats{level}.pt"))
                    counter = torch.load(os.path.join(dino_dir, f"{time_idx}_counter{level}.pt"))
                    if featss is None:
                        featss = (feats/(counter + 1e-16)).view(-1, feats.shape[-1])[::100]
                    else:
                        featss = torch.cat([featss,  (feats/(counter + 1e-16)).view(-1, feats.shape[-1])[::100]], dim=0)
                    time_idx += 1
                pca = PCA(n_components=3).fit(featss)
                featss = None

                time_idx = 0
                grids = None
                while os.path.exists(os.path.join(dino_dir, f"{time_idx}_feats{level}.pt")):
                    
                    feats = torch.load(os.path.join(dino_dir, f"{time_idx}_feats{level}.pt"))
                    sals = torch.load(os.path.join(dino_dir, f"{time_idx}_sals{level}.pt"))
                    counter = torch.load(os.path.join(dino_dir, f"{time_idx}_counter{level}.pt"))
                    #assert False, [feats.shape, sals.shape, counter.shape]
                
                    feats /= counter + 1e-16
                    sals /= counter + 1e-16
                    
                    old_shape = feats.shape
                    feats = torch.from_numpy(pca.transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], 3)
            
                    for comp_idx in range(3):
                        comp = feats[..., comp_idx]
                        comp_min = torch.min(comp)
                        comp_max = torch.max(comp)
                        comp = (comp - comp_min) / (comp_max - comp_min)
                        feats[..., comp_idx] = comp
                    #assert False, [feats.shape, sals.shape]
                    #assert False, torchvision.transforms.functional.to_pil_image(feats[0].permute(2, 0, 1)).save("test.png")
                    grid = make_grid([feats.permute(2, 0, 1)*255., sals.permute(2, 0, 1).repeat(3, 1, 1)*255.])
                    #assert False, [feat.shape, sal.shape, grid.shape]
                    #assert False, grid.shape
                    if grids is None:
                        grids = grid.permute(1, 2, 0)[None, ...]
                    else:
                        grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
                    time_idx += 1
        
                torchvision.io.write_video(os.path.join(dino_dir, f"{level}_.mp4"), 
                grids, fps=1)
            
            time_idx = 0
            grids = None
            featss = None
            while os.path.exists(os.path.join(dino_dir, f"{time_idx}_feats.pt")):
                feats = torch.load(os.path.join(dino_dir, f"{time_idx}_feats.pt")) 
                sals = torch.load(os.path.join(dino_dir, f"{time_idx}_sals.pt"))
                counter = torch.load(os.path.join(dino_dir, f"{time_idx}_counter.pt")) 
                for level in ["_1", "_2"]:
                    feats += torch.nn.functional.interpolate(torch.load(os.path.join(dino_dir, f"{time_idx}_feats{level}.pt")).permute(2, 0, 1)[None, ...], (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0)
                    sals += torch.nn.functional.interpolate(torch.load(os.path.join(dino_dir, f"{time_idx}_sals{level}.pt")).permute(2, 0, 1)[None, ...], (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0)
                    counter += torch.nn.functional.interpolate(torch.load(os.path.join(dino_dir, f"{time_idx}_counter{level}.pt")).permute(2, 0, 1)[None, ...], (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0)
                
                feats /= counter + 1e-16
                sals /= counter + 1e-16
                if featss is None:
                    featss = feats[None, ...]
                    salss = sals[None, ...]
                else:
                    featss = torch.cat([featss, feats[None, ...]], dim=0)
                    salss = torch.cat([salss, sals[None, ...]], dim=0)
                time_idx += 1
            feats = featss
            sals = salss
            old_shape = feats.shape
            feats = torch.nn.functional.normalize(feats, p=2, eps=1e-12, dim=-1)
            pca = PCA(n_components=64).fit(feats.view(-1, feats.shape[-1])[:100])
            feats = torch.from_numpy(pca.transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], -1)
            pca_2 = PCA(n_components=3).fit(feats.view(-1, feats.shape[-1]))
            feats = torch.from_numpy(pca_2.transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], -1)
                
            for comp_idx in range(3):
                comp = feats[..., comp_idx]
                comp_min = torch.min(comp)
                comp_max = torch.max(comp)
                comp = (comp - comp_min) / (comp_max - comp_min)
                feats[..., comp_idx] = comp
            time_idx = 0
            featss = feats
            salss = sals
            while time_idx < featss.shape[0]:
                feats = featss[time_idx]
                sals = salss[time_idx]    
                grid = make_grid([feats.permute(2, 0, 1)*255., sals.permute(2, 0, 1).repeat(3, 1, 1)*255.])
                #assert False, [feat.shape, sal.shape, grid.shape]
                #assert False, grid.shape
                if grids is None:
                    grids = grid.permute(1, 2, 0)[None, ...]
                else:
                    grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
                time_idx += 1
                print(time_idx)
    
            torchvision.io.write_video(os.path.join(dino_dir, f"merged.mp4"), 
            grids, fps=1)
            print(f"Done with {dino_dir}")
            '''
            time_idx = 0
            grids = None
            featss = None
            
            weights = [1/3., 1/3., 1/3.]
            
            while os.path.exists(os.path.join(dino_dir, f"{time_idx}_feats.pt")):
                feats = torch.load(os.path.join(dino_dir, f"{time_idx}_feats.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter.pt")))
                feats_1 = torch.load(os.path.join(dino_dir, f"{time_idx}_feats_1.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter_1.pt")))
                feats_2 = torch.load(os.path.join(dino_dir, f"{time_idx}_feats_2.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter_2.pt")))
                #sals = torch.load(os.path.join(dino_dir, f"{time_idx}_sals.pt"))
                #counter = 
                #for level in ["_1", "_2"]:
                #    feats += torch.nn.functional.interpolate(torch.load(os.path.join(dino_dir, f"{time_idx}_feats{level}.pt")).permute(2, 0, 1)[None, ...], (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0)
                #    sals += torch.nn.functional.interpolate(torch.load(os.path.join(dino_dir, f"{time_idx}_sals{level}.pt")).permute(2, 0, 1)[None, ...], (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0)
                #    counter += torch.nn.functional.interpolate(torch.load(os.path.join(dino_dir, f"{time_idx}_counter{level}.pt")).permute(2, 0, 1)[None, ...], (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0)
                feats = weights[0] * feats +\
                    weights[1]*torch.nn.functional.interpolate(feats_1[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) + \
                    weights[2] *torch.nn.functional.interpolate(feats_2[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) 
                #feats /= counter + 1e-16
                #sals /= counter + 1e-16
                #feats = feats.view(-1, feats.shape[-1])[::100]
                if featss is None:
                    featss = feats.view(-1, feats.shape[-1])[::100]
                    #featss_1 = feats_1.view(-1, feats.shape[-1])[::100]
                    #featss_2 = feats_2.view(-1, feats.shape[-1])[::100]
                else:
                    featss = torch.cat([featss, feats.view(-1, feats.shape[-1])[::100]], dim=0)
                    #featss_1 = torch.cat([featss_1, feats_1.view(-1, feats.shape[-1])[::100]], dim=0)
                    #featss_2 = torch.cat([featss_2, feats_2.view(-1, feats.shape[-1])[::100]], dim=0)
                time_idx += 1
            #feats_1 = featss_1
            #feats_2 = featss_2
            def pca_transform(feats):
                old_shape = feats.shape
                feats = torch.nn.functional.normalize(feats, p=2, eps=1e-12, dim=-1)
                pca = PCA(n_components=64).fit(feats.view(-1, feats.shape[-1]))
                return pca
                #feats = torch.from_numpy(pca.transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], -1)
                #pca_2 = PCA(n_components=3).fit(feats.view(-1, feats.shape[-1]))
                #return pca, pca_2
                #feats = torch.from_numpy(pca_2.transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], -1)
                    
                #for comp_idx in range(3):
                #    comp = feats[..., comp_idx]
                #    comp_min = torch.min(comp)
                #    comp_max = torch.max(comp)
                #    comp = (comp - comp_min) / (comp_max - comp_min)
                #    feats[..., comp_idx] = comp
                #return feats
            pca = pca_transform(featss)
            #pca_1 = pca_transform(feats_1)
            #pca_2 = pca_transform(feats_2)
            
            
            #feats = weights[0] * feats \
            #+ weights[1] * torch.nn.functional.interpolate(feats_1.permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(0, 2, 3, 1)\
            #+ weights[2] * torch.nn.functional.interpolate(feats_2.permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(0, 2, 3, 1)
            
            time_idx = 0
            featss = None
            #salss = sals
            while os.path.exists(os.path.join(dino_dir, f"{time_idx}_feats.pt")):
                def load_feat(feats, pca):
                    old_shape = feats.shape
                    feats = torch.nn.functional.normalize(feats, p=2, eps=1e-12, dim=-1)
                    feats = torch.from_numpy(pca.transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], -1)
                    #feats = torch.nn.functional.normalize(feats, p=2, eps=1e-12, dim=-1)
                    return feats
                feats = torch.load(os.path.join(dino_dir, f"{time_idx}_feats.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter.pt")))
                feats_1 = torch.load(os.path.join(dino_dir, f"{time_idx}_feats_1.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter_1.pt")))
                feats_2 = torch.load(os.path.join(dino_dir, f"{time_idx}_feats_2.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter_2.pt")))
                
                feats = weights[0] * feats +\
                 weights[1] * torch.nn.functional.interpolate(feats_1[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) +\
                 weights[2] * torch.nn.functional.interpolate(feats_2[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) 


                feats = load_feat(feats, pca)
                #feats_1 = load_feat(feats_1, pca_1)
                #feats_2 = load_feat(feats_2, pca_2)
                
                if featss is None:
                    featss = feats[None, ...]
                else:
                    featss = torch.cat([featss, feats[None, ...]], dim=0)
                time_idx += 1
                print(time_idx)
            old_shape = featss.shape
            cpca = PCA(n_components=3).fit(featss.view(-1, featss.shape[-1]))
            featss = torch.from_numpy(cpca.transform(featss.view(-1, featss.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], -1)
            for comp_idx in range(3):
                comp = featss[..., comp_idx]
                comp_min = torch.min(comp)
                comp_max = torch.max(comp)
                comp = (comp - comp_min) / (comp_max - comp_min)
                featss[..., comp_idx] = comp
            time_idx = 0
            #salss = sals
            while os.path.exists(os.path.join(dino_dir, f"{time_idx}_feats.pt")):
                feats = featss[time_idx]
                #sals = salss[time_idx]    
                grid = make_grid([feats.permute(2, 0, 1)*255.])
                #assert False, [feat.shape, sal.shape, grid.shape]
                #assert False, grid.shape
                if grids is None:
                    grids = grid.permute(1, 2, 0)[None, ...]
                else:
                    grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
                time_idx += 1
                print(time_idx)
            if grids.shape[-2] % 2 == 1:
                grids = grids[:, :, :-1, :]
            torchvision.io.write_video(os.path.join(dino_dir, f"weighted.mp4"), 
            grids, fps=1)
            print(f"Done with {dino_dir}")
        