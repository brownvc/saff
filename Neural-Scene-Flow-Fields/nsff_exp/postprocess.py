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
    scenes = ["DynamicFace-2", "Truck-2","Umbrella", "Balloon1-2", "Balloon2-2", "playground", "Jumping", "Skating-2", ]
    root_dir = "../../data/ours_1018"
    out_dir = root_dir + "_processed"
    for scene in tqdm(scenes):
        
        assert os.path.exists(os.path.join(root_dir, scene))
        os.makedirs(os.path.join(out_dir, scene), exist_ok=True)
        image_id = 0
        while os.path.exists(os.path.join(root_dir, scene, f"{image_id}.png")):
            img = imread(os.path.join(root_dir, scene, f"{image_id}.png"))
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
            imsave(os.path.join(out_dir, scene, f"{image_id}_label.png"), label_im)
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
            imsave(os.path.join(out_dir, scene, f"{image_id}_small.png"), to_collapse)

            #img[to_collapse] = 
            # translate back to colored cluster image
            img = old_img
            tmp = np.zeros((img.shape[0], img.shape[1], 3))
            for color, idx in zip(unique_colors, ids):
                tmp[img == idx] = color
            tmp[to_collapse, :] *= 0 
            #assert False, imsave("test.png", tmp)
            imsave(os.path.join(out_dir, scene, f"{image_id}.png"), tmp)
            image_id += 1

