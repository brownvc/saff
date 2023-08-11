import argparse
import numpy as np
import cv2
import os
import json
from tqdm import tqdm

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--raw_folder", type=str, default="../../data/no_sal")
    parser.add_argument("--gt_folder", type=str, default="../../data/gt_masks")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--flip_fg", action="store_true")
    #parser.add_argument("--out_folder", type=str, default="")
    #parser.add_argument("--compact_rgb", type=str, default="20")
    #parser.add_argument("--sdim_depth", type=str, default="40")
    #parser.add_argument("--sdim_rgb", type=str, default="20")
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    #scenes = [ "Skating-2", "Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Truck-2", "Umbrella"]
    out_folder = os.path.join(args.raw_folder, "oracle")
    
    os.makedirs(out_folder, exist_ok=True)
    # find the background colors based on first frame
    gt_img = cv2.imread(os.path.join(args.gt_folder, "00000.png.png"), cv2.IMREAD_GRAYSCALE)
    raw_img = cv2.imread(os.path.join(args.raw_folder, "0.png"))
    #assert False, [gt_img.shape, raw_img.shape]
    is_foreground = gt_img < 10
    if args.flip_fg:
        is_foreground = gt_img > 10
    #cv2.imwrite("test.png", is_foreground*255.)
    #assert False, "Pause"
    unique_colors = np.unique(raw_img.reshape((-1, 3)), axis=0)
    #assert False, [is_foreground.shape, unique_colors]
    #print(unique_colors)
    background = []
    for color in unique_colors:
        region = (raw_img[..., 0:1] == color[0]) & (raw_img[..., 1:2] == color[1])& (raw_img[..., 2:3] == color[2])
        #print(color, region.shape)
        #cv2.imwrite(f"test_{color}.png", raw_img)
        #input()
        #assert False, region.shape
        region = region[..., 0]
        ratio = np.sum(region & is_foreground) / np.sum(region).astype(float)
        print(color, ratio)
        if ratio < args.threshold:
            background.append(color)
    
    idx = 0
    while os.path.exists(os.path.join(args.raw_folder, f"{idx}.png")):
        raw_img = cv2.imread(os.path.join(args.raw_folder, f"{idx}.png"))
        for color in background:
            region = (raw_img[..., 0:1] == color[0]) & (raw_img[..., 1:2] == color[1])& (raw_img[..., 2:3] == color[2])            
            raw_img[region[..., 0]] *= 0
        cv2.imwrite(os.path.join(out_folder, f"{idx}.png"), raw_img)
        idx += 1