import argparse
import numpy as np
import cv2
from metrics import *
import os
import torch
import json

if __name__ == '__main__':
    
    vis_folder = "/mnt/d/Research/NOF/data/VIS_results"
    gt_folder = "/mnt/d/Research/NOF/data/gt_masks"

    result = {
        
        }
    scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "umbrella"]
    for scene in scenes:
        #print(os.path.isdir(os.path.join(gt_folder, scene)))
       
        result[scene] = {
            "training": {
                "ARI": [],
                "fg-ARI": []
            },
            "nv_spatial": {
                "ARI": [],
                "fg-ARI": []
            },
            "nv_static": {
                "ARI": [],
                "fg-ARI": []
            }
        }
        # training images
        with open(os.path.join(vis_folder, f"{scene}_train.npy"), 'rb') as f:
            pred_masks = np.load(f)
            #assert False, [pred_masks.shape, np.unique(pred_masks)]
        for num in range(24):
            gt_mask = cv2.imread(os.path.join(gt_folder, scene, "training_%2d.png" % num), cv2.IMREAD_GRAYSCALE)
            fg_mask = np.zeros(gt_mask.shape).astype(bool)
            fg_mask[gt_mask != 0] = True
            if num == 0:
                pred_masks = torch.from_numpy(pred_masks)[:, None, :, :]
                pred_masks = torch.nn.functional.interpolate(pred_masks, size=(gt_mask.shape[0], gt_mask.shape[1]), mode='nearest').numpy()[:, 0, :, :]
            pred_mask = pred_masks[num]
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            result[scene]["training"]["ARI"].append(ARI(gt_mask, pred_mask))
            result[scene]["training"]["fg-ARI"].append(ARI(gt_mask, pred_mask, fg_mask))
        result[scene]["training"]["mean-ARI"] = sum(result[scene]["training"]["ARI"]) / float(len(result[scene]["training"]["ARI"]))
        result[scene]["training"]["mean-fg-ARI"] = sum(result[scene]["training"]["fg-ARI"]) / float(len(result[scene]["training"]["fg-ARI"]))
        #assert False
        # nv_spatial
        with open(os.path.join(vis_folder, f"{scene}_nv_spatial.npy"), 'rb') as f:
            pred_masks = np.load(f)
        for num in range(25, 48):
            gt_mask = cv2.imread(os.path.join(gt_folder, scene, "nv_spatial_%2d.png" % num), cv2.IMREAD_GRAYSCALE)
            fg_mask = np.zeros(gt_mask.shape).astype(bool)
            fg_mask[gt_mask != 0] = True
            if num == 25:
                pred_masks = torch.from_numpy(pred_masks)[:, None, :, :]
                pred_masks = torch.nn.functional.interpolate(pred_masks, size=(gt_mask.shape[0], gt_mask.shape[1]), mode='nearest').numpy()[:, 0, :, :]
            pred_mask = pred_masks[num-25]
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            result[scene]["nv_spatial"]["ARI"].append(ARI(gt_mask, pred_mask))
            result[scene]["nv_spatial"]["fg-ARI"].append(ARI(gt_mask, pred_mask, fg_mask))
        result[scene]["nv_spatial"]["mean-ARI"] = sum(result[scene]["nv_spatial"]["ARI"]) / float(len(result[scene]["nv_spatial"]["ARI"]))
        result[scene]["nv_spatial"]["mean-fg-ARI"] = sum(result[scene]["nv_spatial"]["fg-ARI"]) / float(len(result[scene]["nv_spatial"]["fg-ARI"]))


        # nv_static
        with open(os.path.join(vis_folder, f"{scene}_nv_static.npy"), 'rb') as f:
            pred_masks = np.load(f)
        for num in range(49, 60):
            gt_mask = cv2.imread(os.path.join(gt_folder, scene, "nv_static_%2d.png" % num), cv2.IMREAD_GRAYSCALE)
            fg_mask = np.zeros(gt_mask.shape).astype(bool)
            fg_mask[gt_mask != 0] = True
            if num == 49:
                pred_masks = torch.from_numpy(pred_masks)[:, None, :, :]
                pred_masks = torch.nn.functional.interpolate(pred_masks, size=(gt_mask.shape[0], gt_mask.shape[1]), mode='nearest').numpy()[:, 0, :, :]
            pred_mask = pred_masks[num-49]
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            result[scene]["nv_static"]["ARI"].append(ARI(gt_mask, pred_mask))
            result[scene]["nv_static"]["fg-ARI"].append(ARI(gt_mask, pred_mask, fg_mask))
        result[scene]["nv_static"]["mean-ARI"] = sum(result[scene]["nv_static"]["ARI"]) / float(len(result[scene]["nv_static"]["ARI"]))
        result[scene]["nv_static"]["mean-fg-ARI"] = sum(result[scene]["nv_static"]["fg-ARI"]) / float(len(result[scene]["nv_static"]["fg-ARI"]))
    result["mean"] = {
            "training": {
                "ARI": [],
                "fg-ARI": []
            },
            "nv_spatial": {
                "ARI": [],
                "fg-ARI": []
            },
            "nv_static": {
                "ARI": [],
                "fg-ARI": []
            }
        }
    
    result["mean"]["training"]["ARI"] = sum([result[scene]["training"]["mean-ARI"] for scene in scenes]) / float(len(scenes))
    result["mean"]["training"]["fg-ARI"] = sum([result[scene]["training"]["mean-fg-ARI"] for scene in scenes]) / float(len(scenes))
    result["mean"]["nv_spatial"]["ARI"] = sum([result[scene]["nv_spatial"]["mean-ARI"] for scene in scenes]) / float(len(scenes))
    result["mean"]["nv_spatial"]["fg-ARI"] = sum([result[scene]["nv_spatial"]["mean-fg-ARI"] for scene in scenes]) / float(len(scenes))
    result["mean"]["nv_static"]["ARI"] = sum([result[scene]["nv_static"]["mean-ARI"] for scene in scenes]) / float(len(scenes))
    result["mean"]["nv_static"]["fg-ARI"] = sum([result[scene]["nv_static"]["mean-fg-ARI"] for scene in scenes]) / float(len(scenes))
    with open(os.path.join(vis_folder, "VIS_result.json"), 'w') as f:
        json.dump(result, f, indent=4)
    