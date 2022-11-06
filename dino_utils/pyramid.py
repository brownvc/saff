import torch
import torchvision
import torch.nn.functional as F
from extractor import *
from cosegmentation import *
from sklearn.decomposition import PCA

import os
import numpy as np
import imageio
#import imageio.v3 as iio

from PIL import Image

def load_feat_sal(dino_dir, n_components, weights, sal_weights, pca=None):
    time_idx = 0
    if pca is None:
        featss = None
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
                weights[1] * torch.nn.functional.interpolate(feats_1[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) + \
                weights[2] * torch.nn.functional.interpolate(feats_2[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) 
            
            #
            
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
            pca = PCA(n_components=n_components).fit(feats.view(-1, feats.shape[-1]))
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
        
        sals = torch.load(os.path.join(dino_dir, f"{time_idx}_sals.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter.pt")))
        sals_1 = torch.load(os.path.join(dino_dir, f"{time_idx}_sals_1.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter_1.pt")))
        sals_2 = torch.load(os.path.join(dino_dir, f"{time_idx}_sals_2.pt")) / (1e-16+torch.load(os.path.join(dino_dir, f"{time_idx}_counter_2.pt")))

        feats = weights[0] * feats +\
            weights[1] * torch.nn.functional.interpolate(feats_1[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) +\
            weights[2] * torch.nn.functional.interpolate(feats_2[None, ...].permute(0, 3, 1, 2), (feats.shape[0], feats.shape[1]), mode='nearest')[0].permute(1, 2, 0) 

        sals = sal_weights[0] * sals +\
            sal_weights[1] * torch.nn.functional.interpolate(sals_1[None, ...].permute(0, 3, 1, 2), (sals.shape[0], sals.shape[1]), mode='nearest')[0].permute(1, 2, 0) + \
            sal_weights[2] * torch.nn.functional.interpolate(sals_2[None, ...].permute(0, 3, 1, 2), (sals.shape[0], sals.shape[1]), mode='nearest')[0].permute(1, 2, 0) 

        feats = load_feat(feats, pca)
        #feats_1 = load_feat(feats_1, pca_1)
        #feats_2 = load_feat(feats_2, pca_2)
        
        if featss is None:
            featss = feats[None, ...]
            salss = sals[None, ...]
        else:
            featss = torch.cat([featss, feats[None, ...]], dim=0)
            salss = torch.cat([salss, sals[None, ...]], dim=0)
        time_idx += 1
        print(time_idx)
    return featss, salss, pca

'''
def read_pca(root_dir, n_components):
    
    ret = {
        "pca": None,
        "pca_1": None,
        "pca_2": None,
    }
    # get each level's pca
    for level in ["", "_1", "_2"]:
        time_idx = 0
        featss = None
        while os.path.exists(os.path.join(root_dir, f"{time_idx}_feats{level}.pt")):
            feats = torch.load(os.path.join(root_dir, f"{time_idx}_feats{level}.pt"))
            counter = torch.load(os.path.join(root_dir, f"{time_idx}_counter{level}.pt"))
            if featss is None:
                featss = (feats/(counter + 1e-16)).view(-1, feats.shape[-1])[::100]
            else:
                featss = torch.cat([featss,  (feats/(counter + 1e-16)).view(-1, feats.shape[-1])[::100]], dim=0)
            time_idx += 1
        featss = torch.nn.functional.normalize(featss, dim=-1)
        ret[f"pca{level}"] = PCA(n_components=n_components).fit(featss)
        featss = None
        
    return ret

def load_feat_sal(root_dir, pcas):
    ret = {
        "feats": None,
        "sals": None,
        "feats_1": None,
        "sals_1": None,
        "feats_2": None,
        "sals_2": None
    }
    for level in ["", "_1", "_2"]:
        time_idx = 0
        while os.path.exists(os.path.join(root_dir, f"{time_idx}_feats{level}.pt")):
            feats = torch.load(os.path.join(root_dir, f"{time_idx}_feats{level}.pt"))
            sals = torch.load(os.path.join(root_dir, f"{time_idx}_sals{level}.pt"))
            counter = torch.load(os.path.join(root_dir, f"{time_idx}_counter{level}.pt"))
            

            feats /= (1e-16 + counter)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            sals /= (1e-16 + counter)
            old_shape = feats.shape
            

            if ret[f"feats{level}"] is None:
                ret[f"feats{level}"] = torch.from_numpy(pcas[f"pca{level}"].transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], -1)[None, ...]
                ret[f"sals{level}"] = sals[None, ...]
            else:
                ret[f"feats{level}"] = torch.cat([
                    ret[f"feats{level}"], 
                    torch.from_numpy(pcas[f"pca{level}"].transform(feats.view(-1, feats.shape[-1]).numpy())).view(old_shape[0], old_shape[1], -1)[None, ...]], 
                dim=0)
                ret[f"sals{level}"] = torch.cat([ret[f"sals{level}"], sals[None, ...]], dim=0)
            time_idx += 1

    height, width = ret["feats"].shape[1], ret["feats"].shape[2]    

    for level in ["_1", "_2"]:
        ret[f"feats{level}"] = torch.nn.functional.interpolate(ret[f"feats{level}"].permute(0, 3, 1, 2), (height, width), mode="nearest").permute(0, 2, 3, 1)
        ret[f"sals{level}"] = torch.nn.functional.interpolate(ret[f"sals{level}"].permute(0, 3, 1, 2), (height, width), mode="nearest").permute(0, 2, 3, 1)

    return ret

'''