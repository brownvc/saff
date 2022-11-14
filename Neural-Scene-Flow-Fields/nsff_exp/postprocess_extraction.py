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

#https://medium.com/swlh/image-processing-with-python-connected-components-and-region-labeling-3eef1864b951

square = np.ones((3, 3))
def multi_dil(im, num, element=square):
    for i in range(num):
        im = dilation(im, element)
    return im
def multi_ero(im, num, element=square):
    for i in range(num):
        im = erosion(im, element)
    return im
#def imread(f):
#    if f.endswith('png'):
#        return imageio.imread(f, ignoregamma=True)
#    else:
#        return imageio.imread(f)

if __name__ == "__main__":
    root_dir = "/users/yliang51/data/yliang51/NOF/Neural-Scene-Flow-Fields/nsff_exp/logs/experiment_dynamicFace_sal_multi_F00-30/extract_2D-010_path_360001"
    mask_dir = "/users/yliang51/data/yliang51/NOF/data/ours_1018_processed/DynamicFace-2"
    
    for image_id in tqdm(range(60)):
        rgb = imread(os.path.join(root_dir, f"{image_id}_rgb.png"))
        depth = imread(os.path.join(root_dir, f"{image_id}_depth.png"))
        alpha = cv2.imread(os.path.join(root_dir, f"{image_id}_depth.png"), cv2.IMREAD_GRAYSCALE).astype(float)
        mask = cv2.imread(os.path.join(mask_dir, f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)

        rgb[mask == 0, :] *= 0
        depth[mask == 0] *= 0
        alpha[mask == 0] = 255.
        
        imsave(os.path.join(root_dir, f"{image_id}_rgb_post.png"), rgb)
        imsave(os.path.join(root_dir, f"{image_id}_depth_post.png"), depth)
        cv2.imwrite(os.path.join(root_dir, f"{image_id}_alpha_post.png"), alpha)
        
    assert False

    #assert False, imsave("test.png", img)
    #assert False, img.shape
    #(288, 54x, 3)
    unique_colors = np.unique(img.reshape((-1, 3)), axis=0)[:,None, None, :]
    ids = list(range(len(unique_colors)))
    tmp = np.zeros_like(img).astype(int)-1
    for color, idx in zip(unique_colors, ids):
        #assert False, [pred_mask.shape, color.shape]
        #print(color)
        if color[0][0][0] == 0 and color[0][0][1] == 0 and color[0][0][2] == 0:
            continue
        tmp[img == color] = idx
    img = tmp[..., 0]
    #assert False, imsave("test.png", img*50.)
    '''
    # translate back to colored cluster image
    tmp = np.zeros((img.shape[0], img.shape[1], 3))
    for color, idx in zip(unique_colors, ids):
        tmp[img == idx] = color
    assert False, imsave("test.png", tmp)
    '''
    old_img = copy.deepcopy(img)
    
    
    
    #dilation and erosion
    #img = multi_dil(img, 2)
    #img = area_closing(img, 500)
    img = multi_ero(img, 3)
    img = opening(img)
    img = multi_dil(img, 3)
    img = area_closing(img, 500)
    is_obj = np.zeros_like(img)
    is_obj[img > 0] = 1

    #assert False, imsave("test.png", is_obj)
    
    '''

    label_im = label(img, connectivity=2)
    regions = regionprops(label_im)
    #assert False, imsave("test.png", label_im)
    
    list_of_index = []
    for num, x in enumerate(regions):
        area = np.sum(label_im == num)
        #convex_area = x.convex_area
        if (area>100):
            list_of_index.append(num)
    #print(list_of_index)
    to_collapse = ~np.isin(label_im, list_of_index) 
    #assert False, imsave("test.png", to_collapse)
    '''
    
    img = old_img
    img[is_obj == 0] *= 0
    old_img = copy.deepcopy(img)
    label_im = label(img, connectivity=2)
    #regions = regionprops(label_im)
    list_of_index = []
    for num, x in enumerate(np.unique(label_im)):
        area = np.sum(label_im == x)
        #convex_area = x.convex_area
        #print(num, x,)
        if area > 0.01*img.shape[0]*img.shape[1]:
            list_of_index.append(num)
    #print(list_of_index)
    #assert False, np.unique(label_im)
    to_collapse = ~np.isin(label_im, list_of_index) 

    #img[to_collapse] = 
    # translate back to colored cluster image
    img = old_img
    tmp = np.zeros((img.shape[0], img.shape[1], 3))
    for color, idx in zip(unique_colors, ids):
        tmp[img == idx] = color
    tmp[to_collapse, :] *= 0 
    #assert False, imsave("test.png", tmp)
    imsave("out.png", tmp)
    

