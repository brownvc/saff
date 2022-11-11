import argparse
import numpy as np
import cv2
from metrics import *
import os
import torch
import json

if __name__ == '__main__':
    
    vis_folder = "../data/dino_masks"
    gt_folder = "../data/gt_masks"

    result = {
        
        }
    scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Umbrella"]
    
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
        #with open(os.path.join(vis_folder, f"{scene}_train.npy"), 'rb') as f:
        #    pred_masks = np.load(f)
            #assert False, [pred_masks.shape, np.unique(pred_masks)]
        #dynamicFace-2: missing first frame

        #assert False
        # nv_spatial
        #with open(os.path.join(vis_folder, f"{scene}_nv_spatial.npy"), 'rb') as f:
        #    pred_masks = np.load(f)
        #0, 2, ... 23
        for num in range(24):
            gt_mask = cv2.imread(os.path.join(gt_folder, scene, "training_%2d.png" % num), cv2.IMREAD_GRAYSCALE)
            gt_mask = gt_mask.astype(int)
            gt_mask[gt_mask == 0] = -1
            
            pred_mask = cv2.imread(os.path.join(vis_folder, scene, "train", f"{num}.png"))
            unique_colors = np.unique(pred_mask.reshape((-1, 3)), axis=0)[:,None, None, :]
            ids = list(range(len(unique_colors)))
            tmp = np.zeros_like(pred_mask).astype(int)-1
            for color, idx in zip(unique_colors, ids):
                #assert False, [pred_mask.shape, color.shape]
                #print(color)
                if color[0][0][0] == 0 and color[0][0][1] == 0 and color[0][0][2] == 0:
                    continue
                tmp[pred_mask == color] = idx
            pred_mask = tmp[..., 0]
            
            #assert False, np.unique(pred_mask)
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            #assert False, [np.unique(pred_mask), np.unique(gt_mask)]
            result[scene]["training"]["Jaccard"].append(compute_jaccard(gt_mask, pred_mask))
            #if not os.path.exists(os.path.join(vis_folder, scene, "train", "mask_rgb_%06d.png" % (num+2))):
            #    break
        result[scene]["training"]["mean-Jaccard"] = sum(result[scene]["training"]["Jaccard"]) / float(len(result[scene]["training"]["Jaccard"]))
        

        for num in range(25, 48):
            gt_mask = cv2.imread(os.path.join(gt_folder, scene, "nv_spatial_%2d.png" % num), cv2.IMREAD_GRAYSCALE)
            gt_mask = gt_mask.astype(int)
            gt_mask[gt_mask == 0] = -1
            
            pred_mask = cv2.imread(os.path.join(vis_folder, scene, "nv_spatial", f"{num-24}.png"))
            unique_colors = np.unique(pred_mask.reshape((-1, 3)), axis=0)[:,None, None, :]
            ids = list(range(len(unique_colors)))
            tmp = np.zeros_like(pred_mask).astype(int)-1
            for color, idx in zip(unique_colors, ids):
                #assert False, [pred_mask.shape, color.shape]
                #print(color)
                if color[0][0][0] == 0 and color[0][0][1] == 0 and color[0][0][2] == 0:
                    continue
                tmp[pred_mask == color] = idx
            pred_mask = tmp[..., 0]
            
            #assert False, np.unique(pred_mask)
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            #assert False, [np.unique(pred_mask), np.unique(gt_mask)]
            result[scene]["nv_spatial"]["Jaccard"].append(compute_jaccard(gt_mask, pred_mask))
            #if not os.path.exists(os.path.join(vis_folder, scene, "train", "mask_rgb_%06d.png" % (num+2))):
            #    break
        result[scene]["nv_spatial"]["mean-Jaccard"] = sum(result[scene]["nv_spatial"]["Jaccard"]) / float(len(result[scene]["nv_spatial"]["Jaccard"]))
        
        for num in range(49, 60):
            gt_mask = cv2.imread(os.path.join(gt_folder, scene, "nv_static_%2d.png" % num), cv2.IMREAD_GRAYSCALE)
            gt_mask = gt_mask.astype(int)
            gt_mask[gt_mask == 0] = -1
            
            pred_mask = cv2.imread(os.path.join(vis_folder, scene, "nv_static", f"{num-48}.png"))
            unique_colors = np.unique(pred_mask.reshape((-1, 3)), axis=0)[:,None, None, :]
            ids = list(range(len(unique_colors)))
            tmp = np.zeros_like(pred_mask).astype(int)-1
            for color, idx in zip(unique_colors, ids):
                #assert False, [pred_mask.shape, color.shape]
                #print(color)
                if color[0][0][0] == 0 and color[0][0][1] == 0 and color[0][0][2] == 0:
                    continue
                tmp[pred_mask == color] = idx
            pred_mask = tmp[..., 0]
            
            #assert False, np.unique(pred_mask)
            #assert False, [gt_mask.shape, fg_mask.shape, pred_mask.shape]
            #print(num)
            #print("ARI: ", ARI(gt_mask, pred_mask))
            #print("fg-ARI: ", ARI(gt_mask, pred_mask, fg_mask))
            #assert False, [np.unique(pred_mask), np.unique(gt_mask)]
            result[scene]["nv_static"]["Jaccard"].append(compute_jaccard(gt_mask, pred_mask))
            #if not os.path.exists(os.path.join(vis_folder, scene, "train", "mask_rgb_%06d.png" % (num+2))):
            #    break
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
    