import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
import os
import copy
from tqdm import tqdm
import cv2

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


#https://medium.com/swlh/image-processing-with-python-connected-components-and-region-labeling-3eef1864b951


#def imread(f):
#    if f.endswith('png'):
#        return imageio.imread(f, ignoregamma=True)
#    else:
#        return imageio.imread(f)

if __name__ == "__main__":
    scenes = ["Umbrella", "Skating-2", "DynamicFace-2", "Truck-2", "Balloon1-2", 
    "Balloon2-2", "playground", "Jumping",]
    render_map = {
        "Balloon1-2": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_balloon1-2_2_multi_F00-30/render_2D-010_path_360001",
        "Balloon2-2": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_Balloon2-2_2_multi_F00-30/render_2D-010_path_360001",
        "DynamicFace-2": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_dynamicFace_sal_multi_F00-30/render_2D-010_path_360001",
        "Jumping": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_jumping_sal_multi_F00-30/render_2D-010_path_360001",
        "playground": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_playground_sal_multi_F00-30/render_2D-010_path_360001",
        "Skating-2": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_skating_sal_multi_F00-30/render_2D-010_path_360001",
        "Truck-2": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_truck_sal_multi_F00-30/render_2D-010_path_360001",
        "Umbrella": "../../Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_Umbrella_sal_multi_F00-30/render_2D-010_path_360001"
        }
    root_dir = "../../data/ours_1018"
    out_dir = root_dir + "_processed_crf"
    for scene in tqdm(scenes):
        
        assert os.path.exists(os.path.join(root_dir, scene))
        os.makedirs(os.path.join(out_dir, scene), exist_ok=True)
        image_id = 0
        while os.path.exists(os.path.join(root_dir, scene, f"{image_id}.png")):
            rgb_img = cv2.imread(os.path.join(render_map[scene], f"{image_id}_rgb.png"))
            depth_img = cv2.imread(os.path.join(render_map[scene], f"{image_id}_depth.png"))
            img = cv2.imread(os.path.join(root_dir, scene, f"{image_id}.png"))
            #assert False, imsave("test.png", img)
            #assert False, img.shape
            #(288, 54x, 3)
            unique_colors = np.unique(img.reshape((-1, 3)), axis=0)[:,:]
            #assert False, [unique_colors, unique_colors.dtype, unique_colors.shape]
            U = cv2.imread(os.path.join(root_dir, scene, f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
            labels = np.unique(U)
            mydict = {}
            for i in range(len(labels)):
                mydict[labels[i]] = i
            U = np.vectorize(mydict.get)(U)
            n_labels = np.max(U) + 1
            HAS_UNK = False
            U = unary_from_labels(U, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
            d = dcrf.DenseCRF2D(depth_img.shape[1], depth_img.shape[0], U.shape[0])
            d.setUnaryEnergy(U)

            feats = create_pairwise_gaussian(sdims=(3, 3), shape=depth_img.shape[:2])
            d.addPairwiseEnergy(feats, compat=15,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
            feats = create_pairwise_bilateral(sdims=(40, 40), schan=(13, 13, 13),
                        img=depth_img, chdim=2)
            d.addPairwiseEnergy(feats, compat=30,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
            feats = create_pairwise_bilateral(sdims=(20, 20), schan=(13, 13, 13),
                        img=rgb_img, chdim=2)
            d.addPairwiseEnergy(feats, compat=20,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

            Q = d.inference(5)
            
            #ids = list(range(len(unique_colors)))
            #tmp = np.zeros_like(img).astype(int)-1
            #for color, idx in zip(unique_colors, ids):
                #assert False, [pred_mask.shape, color.shape]
                #print(color)
            #    if color[0][0][0] == 0 and color[0][0][1] == 0 and color[0][0][2] == 0:
            #        continue
            #    tmp[img == color] = idx
            #img = tmp[..., 0]
            #assert False, imsave("test.png", img*50.)
           
            cv2.imwrite(os.path.join(out_dir, scene, f"{image_id}.png"), unique_colors[np.argmax(Q, axis=0), :].reshape(depth_img.shape))
            image_id += 1

