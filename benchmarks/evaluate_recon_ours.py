import argparse
import numpy as np
import cv2
from metrics import *
import os
import torch
import json

if __name__ == '__main__':
    
    vis_folder = "/users/yliang51/data/yliang51/NOF/data/nsff_norig"
    gt_folder = "/users/yliang51/data/yliang51/NOF/data/nvidia_data_full"

    result = {
        
        }
    scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "Umbrella"]
    #scenes = scenes[:2]
    for scene in scenes:
        #print(os.path.isdir(os.path.join(gt_folder, scene)))
       
        result[scene] = {
            "training": {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": []
            },
            "nv_spatial": {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": []
            },
            "nv_static": {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": []
            }
        }
        # training images
        #with open(os.path.join(vis_folder, f"{scene}_train.npy"), 'rb') as f:
        #    pred_masks = np.load(f)
            #assert False, [pred_masks.shape, np.unique(pred_masks)]
        for num in range(24):
            #assert False, os.path.join(gt_folder, scene, "dense", "mv_images", "%05d" % num, "cam%02d.jpg" % (num % 12 + 1))
            gt_img = torch.from_numpy(cv2.imread(os.path.join(gt_folder, "Playground" if scene=="playground" else scene, "dense", "mv_images", "%05d" % num, "cam%02d.jpg" % (num % 12+1)))[..., [2, 1, 0]]/ 255.).cuda().permute(2, 0, 1)[None, ...]
            
            try:
                pred_img = torch.from_numpy(cv2.imread(os.path.join(vis_folder, scene, f"{num}_rgb.png"))[..., [2, 1, 0]]/ 255.).cuda().permute(2, 0, 1)[None, ...]
            except:
                assert False, os.path.join(vis_folder, scene, f"{num}_rgb.png")
            gt_img = torch.nn.functional.interpolate(gt_img, (pred_img.shape[-2], pred_img.shape[-1]), mode="nearest")
            #assert False, [gt_img.shape, pred_img.shape]
            result[scene]["training"]["LPIPS"].append(loss_lpips(gt_img, pred_img))
            result[scene]["training"]["PSNR"].append(loss_psnr(gt_img, pred_img))
            result[scene]["training"]["SSIM"].append(loss_ssim(gt_img, pred_img))
        result[scene]["training"]["mean-LPIPS"] = sum(result[scene]["training"]["LPIPS"]) / float(len(result[scene]["training"]["LPIPS"]))
        result[scene]["training"]["mean-PSNR"] = sum(result[scene]["training"]["PSNR"]) / float(len(result[scene]["training"]["PSNR"]))
        result[scene]["training"]["mean-SSIM"] = sum(result[scene]["training"]["SSIM"]) / float(len(result[scene]["training"]["SSIM"]))

        #assert False
        # nv_spatial
        #with open(os.path.join(vis_folder, f"{scene}_nv_spatial.npy"), 'rb') as f:
        #    pred_masks = np.load(f)
        for num in range(25, 48):
            #assert False, os.path.join(gt_folder, scene, "dense", "mv_images", "%05d" % num, "cam01.jpg")
            gt_img = torch.from_numpy(cv2.imread(os.path.join(gt_folder, "Playground" if scene=="playground" else scene, "dense", "mv_images", "%05d" % (num-25), "cam01.jpg"))[..., [2, 1, 0]]/ 255.).cuda().permute(2, 0, 1)[None, ...]
            pred_img = torch.from_numpy(cv2.imread(os.path.join(vis_folder, scene, f"{num}_rgb.png"))[..., [2, 1, 0]]/ 255.).cuda().permute(2, 0, 1)[None, ...]
            gt_img = torch.nn.functional.interpolate(gt_img, (pred_img.shape[-2], pred_img.shape[-1]), mode="nearest")
            result[scene]["nv_spatial"]["LPIPS"].append(loss_lpips(gt_img, pred_img))
            result[scene]["nv_spatial"]["PSNR"].append(loss_psnr(gt_img, pred_img))
            result[scene]["nv_spatial"]["SSIM"].append(loss_ssim(gt_img, pred_img))
        result[scene]["nv_spatial"]["mean-LPIPS"] = sum(result[scene]["nv_spatial"]["LPIPS"]) / float(len(result[scene]["nv_spatial"]["LPIPS"]))
        result[scene]["nv_spatial"]["mean-PSNR"] = sum(result[scene]["nv_spatial"]["PSNR"]) / float(len(result[scene]["nv_spatial"]["PSNR"]))
        result[scene]["nv_spatial"]["mean-SSIM"] = sum(result[scene]["nv_spatial"]["SSIM"]) / float(len(result[scene]["nv_spatial"]["SSIM"]))
            
        # nv_static
        #with open(os.path.join(vis_folder, f"{scene}_nv_static.npy"), 'rb') as f:
        #    pred_masks = np.load(f)
        for num in range(49, 60):
            gt_img = torch.from_numpy(cv2.imread(os.path.join(gt_folder, "Playground" if scene=="playground" else scene, "dense", "mv_images", "00000", "cam%02d.jpg" % ((num-49) % 12+1)))[..., [2, 1, 0]]/ 255.).cuda().permute(2, 0, 1)[None, ...]
            pred_img = torch.from_numpy(cv2.imread(os.path.join(vis_folder, scene, f"{num}_rgb.png"))[..., [2, 1, 0]]/ 255.).cuda().permute(2, 0, 1)[None, ...]
            gt_img = torch.nn.functional.interpolate(gt_img, (pred_img.shape[-2], pred_img.shape[-1]), mode="nearest")
            result[scene]["nv_static"]["LPIPS"].append(loss_lpips(gt_img, pred_img))
            result[scene]["nv_static"]["PSNR"].append(loss_psnr(gt_img, pred_img))
            result[scene]["nv_static"]["SSIM"].append(loss_ssim(gt_img, pred_img))
        result[scene]["nv_static"]["mean-LPIPS"] = sum(result[scene]["nv_static"]["LPIPS"]) / float(len(result[scene]["nv_static"]["LPIPS"]))
        result[scene]["nv_static"]["mean-PSNR"] = sum(result[scene]["nv_static"]["PSNR"]) / float(len(result[scene]["nv_static"]["PSNR"]))
        result[scene]["nv_static"]["mean-SSIM"] = sum(result[scene]["nv_static"]["SSIM"]) / float(len(result[scene]["nv_static"]["SSIM"]))
            
    result["mean"] = {
            "training": {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": []
            },
            "nv_spatial": {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": []
            },
            "nv_static": {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": []
            }
        }
    
    for split in ["training", "nv_spatial", "nv_static"]:
        for metric in ["PSNR", "SSIM", "LPIPS"]:
            result["mean"][split][metric] = sum([result[scene][split]["mean-"+metric] for scene in scenes])/float(len(scenes))
    
    with open(os.path.join(vis_folder, "ours_recon_result.json"), 'w') as f:
        json.dump(result, f, indent=4)
    