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
                "Jaccard": [],
            },
            "nv_spatial": {
                "Jaccard": [],
            },
            "nv_static": {
                "Jaccard": [],
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
            pred_mask = pred_mask.astype(int)
            gt_mask = gt_mask.astype(int)
            pred_mask[pred_mask == 0] = -1
            gt_mask[gt_mask == 0] = -1
            #assert False, np.unique(pred_mask)
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            #assert False, [np.unique(pred_mask), np.unique(gt_mask)]
            result[scene]["training"]["Jaccard"].append(compute_jaccard(gt_mask, pred_mask))
        result[scene]["training"]["mean-Jaccard"] = sum(result[scene]["training"]["Jaccard"]) / float(len(result[scene]["training"]["Jaccard"]))
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
            pred_mask = pred_mask.astype(int)
            gt_mask = gt_mask.astype(int)
            pred_mask[pred_mask == 0] = -1
            gt_mask[gt_mask == 0] = -1
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            result[scene]["nv_spatial"]["Jaccard"].append(compute_jaccard(gt_mask, pred_mask))
        result[scene]["nv_spatial"]["mean-Jaccard"] = sum(result[scene]["nv_spatial"]["Jaccard"]) / float(len(result[scene]["nv_spatial"]["Jaccard"]))


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
            pred_mask = pred_mask.astype(int)
            gt_mask = gt_mask.astype(int)
            pred_mask[pred_mask == 0] = -1
            gt_mask[gt_mask == 0] = -1
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            result[scene]["nv_static"]["Jaccard"].append(compute_jaccard(gt_mask, pred_mask))
        result[scene]["nv_static"]["mean-Jaccard"] = sum(result[scene]["nv_static"]["Jaccard"]) / float(len(result[scene]["nv_static"]["Jaccard"]))
    result["mean"] = {
             "training": {
                "Jaccard": [],
            },
            "nv_spatial": {
                "Jaccard": [],
            },
            "nv_static": {
                "Jaccard": [],
            }
        }
    
    for split in ["training", "nv_spatial", "nv_static"]:
        for metric in ["Jaccard"]:
            result["mean"][split][metric] = sum([result[scene][split]["mean-"+metric] for scene in scenes])/float(len(scenes))
   
    with open(os.path.join(vis_folder, "j_result.json"), 'w') as f:
        json.dump(result, f, indent=4)
    