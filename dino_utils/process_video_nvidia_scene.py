import torch
import torch.nn.functional as F
from extractor import *
from cosegmentation import *
from sklearn.decomposition import PCA

import os
import numpy as np
import imageio
import imageio.v3 as iio

from PIL import Image

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. small stride -> higher resolution.""")
    parser.add_argument('--load_size', default=128, type=int, help='load size of the input image.')
    parser.add_argument('--model_type', default='dino_vits8', type=str,
    help="""type of model to extract. Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
            vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument("--dino_batch", default=4, type=int, help="""which batch size to prevent explosion""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    
    parser.add_argument("--scene", required=True, type=str)
    
    return parser


def imread(f):
    training_imgs = []
    for idx in range(24):
        img = iio.imread(f, index=idx)
        img = img[..., :3] / 255.
        training_imgs.append(img.astype(np.float32))
    #training_imgs = np.stack(training_imgs, -1)
    nv_spatial_imgs = []
    for idx in range(25, 48):
        img = iio.imread(f, index=idx)
        img = img[..., :3] / 255.
        nv_spatial_imgs.append(img.astype(np.float32))
    nv_static_imgs = []
    for idx in range(49, 60):
        img = iio.imread(f, index=idx)
        img = img[..., :3] / 255.
        nv_static_imgs.append(img.astype(np.float32))
    return training_imgs, nv_spatial_imgs, nv_static_imgs

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    root_dir = "../data/dino_material"
    print("For now, stride=1, and no PCA")
    stride = 1

    '''loading all rgb images'''
    
    print("For now, don't allow resizing images, i.e. factor=None")


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    saliency_extractor = extractor

    dino_batch_size_1 = 16
    dino_batch_size = 8
    print(f"Using dino batch size {dino_batch_size}!")

    scenes = [args.scene]
    
    #for scene in scenes:    
    #    print(os.path.exists(os.path.join(root_dir, f"{scene}.mp4")))   
    #assert False, "All above have to be true to find all video as inputs"
    
    for scene in scenes:
        print(f"Running for scene {scene}:")
        
        os.makedirs(os.path.join(root_dir, scene), exist_ok=True)
        training_imgs, nv_spatial_imgs, nv_static_imgs = imread(os.path.join(root_dir, f"{scene}.mp4"))
        #assert False, [len(training_imgs), len(nv_spatial_imgs),len(nv_static_imgs),
        #training_imgs[0].shape, nv_spatial_imgs[0].shape, nv_static_imgs[0].shape]
        height, width = training_imgs[0].shape[0], training_imgs[0].shape[1]
        #assert False, [height, width, imgs[0].shape]
        if height < width:
            dwidth = int(args.load_size / float(height) * width)
            dheight = args.load_size
        else:
            dheight = int(args.load_size / float(width) * height)
            dwidth = args.load_size

        #for split, imgs in zip(["training", "nv_spatial", "nv_static"], [training_imgs, nv_spatial_imgs, nv_static_imgs]):
        #    outdir = os.path.join(root_dir, scene, split)
        #    print(outdir)
        #assert False, "make sure above directories are correct"

        print("Need to make sure that the windows are precomputed per scene as image shape may vary")
        #assert False, width
        windows = []
        start_height = 0
        while start_height + dheight <= height:
            start_width = 0
            while start_width + dwidth <= width:
                windows.append((start_height, start_width))
                start_width += stride
            start_height += stride
        windows_1 = []
        start_height = 0
        while start_height + dheight <= dheight*2:
            start_width = 0
            while start_width + dwidth <= dwidth*2:
                windows_1.append((start_height, start_width))
                start_width += stride
            start_height += stride
        start_height = 0
        start_width = 0
        #assert False, [len(windows), len(windows_1)]
        

        for split, imgs in zip(["training", "nv_spatial", "nv_static"], [training_imgs, nv_spatial_imgs, nv_static_imgs]):
            outdir = os.path.join(root_dir, scene, split)
            os.makedirs(outdir, exist_ok=True)

            

            feats = None
            feats_1 = None
            feats_2 = None
            
            
            for image_id, img in enumerate(imgs):
                img_1 = Image.fromarray(np.uint8(img*255.)).resize((2*dwidth, 2*dheight), resample=Image.LANCZOS)
                #assert False, img_1.size
                img_2 = img_1.resize((dwidth, dheight), resample=Image.LANCZOS)
                #img_1.save("img_1.png")
                img_1 = np.array(img_1).astype(np.float32)/255.
                #assert False, np.unique(img_1)
                #assert False, img_2.size
                #img_2.save("img_2.png")
                #assert False, "Don't forget to delete above validation saving"
                img_2 = np.array(img_2).astype(np.float32)/255.
                #assert False, np.unique(img_2)

                print(f"Rendering for split {split}, image {image_id}, level 2:")
                if feats_2 is not None:
                    feats_2 = feats_2.to(device)
                    sals_2 = sals_2.to(device)
                with torch.no_grad():
                    batch = torch.from_numpy(img_2).permute(2, 0, 1)[None, ...]
                    feat_raw = extractor.extract_descriptors(batch.to(device), args.layer, args.facet, args.bin)
                    feat_raw = feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)        
                    feat_raw = F.interpolate(feat_raw, size=(dheight, dwidth), mode='nearest')
                    if feats_2 is None:
                        feats_2 = torch.zeros((len(imgs), dheight, dwidth, feat_raw.shape[1])).to(device)
                        sals_2 = torch.zeros((len(imgs), dheight, dwidth, 1)).to(device)     
                        counter_2 = torch.zeros((len(imgs), dheight, dwidth, 1))
                    
                    #assert False, [feats_2.shape, feat_raw.shape]
                    feats_2[image_id:image_id+1] = feat_raw.permute(0, 2, 3, 1).to(device)
                    counter_2[image_id:image_id+1] += 1
                    sal_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                    sal_raw = sal_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                    sal_raw = F.interpolate(sal_raw, size=(dheight, dwidth), mode='nearest')
                    #print(sals_2.device, sal_raw.device)
                    sals_2[image_id:image_id+1] = sal_raw.permute(0, 2, 3, 1).to(device)
                feats_2 = feats_2.cpu()
                sals_2 = sals_2.cpu()

                print(f"Rendering for split {split}, image {image_id}, level 1:")
                #start_height = 0
                if feats_1 is not None:
                    feats_1 = feats_1.to(device)
                    sals_1 = sals_1.to(device)
                pbar = tqdm(total=len(windows_1))
                batch = None
                pos = []
                for start_height, start_width in windows_1:
                    if batch is None:
                        batch = torch.from_numpy(img_1[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]
                        #assert False, batch.shape
                    else:
                        batch = torch.cat([batch, torch.from_numpy(img_1[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]], dim=0)
                        #assert False, batch.shape
                    pos.append((start_height, start_width))
                    if batch.shape[0] >= dino_batch_size_1:
                        with torch.no_grad():
                            feats_raw = extractor.extract_descriptors(batch.to(device), args.layer, args.facet, args.bin)
                            feats_raw = feats_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            feats_raw = F.interpolate(feats_raw, size=(dheight, dwidth), mode='nearest')
                            sals_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                            sals_raw = sals_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            sals_raw = F.interpolate(sals_raw, size=(dheight, dwidth), mode='nearest')
                        if feats_1 is None:
                            feats_1 = torch.zeros((len(imgs), 2*dheight, 2*dwidth, feat_raw.shape[1])).to(device)
                            sals_1 = torch.zeros((len(imgs), 2*dheight, 2*dwidth, 1)).to(device)     
                            counter_1 = torch.zeros((len(imgs), 2*dheight, 2*dwidth, 1))
                        #assert False, [feats_1.shape, feats_raw.shape]
                        for t in range(len(batch)):
                            feat_raw = feats_raw[t:t+1]
                            sal_raw = sals_raw[t:t+1]
                            hstart, wstart = pos[t]
                            feats_1[image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += feat_raw.permute(0, 2, 3, 1)
                            counter_1[image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += 1   
                            sals_1[image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += sal_raw.permute(0, 2, 3, 1)
                        pbar.update(len(batch))
                        batch = None
                        pos = []
                        #break
                        
                feats_1 = feats_1.cpu()
                sals_1 = sals_1.cpu()
                    
                    
                pbar.close()

                

                

                print(f"Rendering for split {split}, image {image_id}, level 0:")
                #start_height = 0
                if feats is not None:
                    feats = feats.to(device)
                    sals = sals.to(device)
                pbar = tqdm(total=len(windows))
                batch = None
                pos = []
                #assert False, img.shape
                for start_height, start_width in windows:
                    #print(start_height, start_width, torch.from_numpy(img[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...].shape)
                    if batch is None:
                        batch = torch.from_numpy(img[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]
                        #assert False, batch.shape
                    else:
                        batch = torch.cat([batch, torch.from_numpy(img[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]], dim=0)
                        #assert False, batch.shape
                    pos.append((start_height, start_width))
                    if batch.shape[0] >= dino_batch_size:
                        with torch.no_grad():
                            feats_raw = extractor.extract_descriptors(batch.to(device), args.layer, args.facet, args.bin)
                            feats_raw = feats_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            feats_raw = F.interpolate(feats_raw, size=(dheight, dwidth), mode='nearest')
                            sals_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                            sals_raw = sals_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            sals_raw = F.interpolate(sals_raw, size=(dheight, dwidth), mode='nearest')
                            if feats is None:
                                feats = torch.zeros((len(imgs), height, width, feat_raw.shape[1])).to(device)
                                sals = torch.zeros((len(imgs), height, width, 1)).to(device)     
                                counter = torch.zeros((len(imgs), height, width, 1))
                            #print(feats.shape, feats_raw.shape)
                            for t in range(len(batch)):
                                feat_raw = feats_raw[t:t+1]
                                sal_raw = sals_raw[t:t+1]
                                hstart, wstart = pos[t]
                                #print(hstart+dheight, wstart+dwidth)
                                feats[image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += feat_raw.permute(0, 2, 3, 1)
                                counter[image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += 1   
                                sals[image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += sal_raw.permute(0, 2, 3, 1)
                        pbar.update(len(batch))
                        batch = None
                        #break
                        pos = []
                        #torch.cuda.empty_cache()
                feats = feats.cpu()
                sals = sals.cpu()
                
                pbar.close()
                #break
                #print("Debuggin! break after first image")
            torch.save(feats, os.path.join(outdir, "feats.pt"))
            torch.save(sals, os.path.join(outdir, "sals.pt"))
            torch.save(counter, os.path.join(outdir, "counter.pt"))

            torch.save(feats_1, os.path.join(outdir, "feats_1.pt"))
            torch.save(sals_1, os.path.join(outdir, "sals_1.pt"))
            torch.save(counter_1, os.path.join(outdir, "counter_1.pt"))

            torch.save(feats_2, os.path.join(outdir, "feats_2.pt"))
            torch.save(sals_2, os.path.join(outdir, "sals_2.pt"))
            torch.save(counter_2, os.path.join(outdir, "counter_2.pt"))
        #break
        #print("Debugging! break after first scene")
    
    '''
    #Visualization
    #divide by counter
    feats /= 1e-16 + counter
    sals /= 1e-16 + counter

    feats_1 /= 1e-16 + counter_1
    sals_1 /= 1e-16 + counter_1

    feats_2 /= 1e-16 + counter_2
    sals_2 /= 1e-16 + counter_2
            
    
    # save features
    Image.fromarray(np.uint8(F.normalize(feats[0, ..., :3], dim=-1).cpu().numpy()* 255.)).save("feat.png")
    Image.fromarray(np.uint8(F.normalize(feats_1[0, ..., :3], dim=-1).cpu().numpy()* 255.)).save("feat_1.png")
    Image.fromarray(np.uint8(F.normalize(feats_2[0, ..., :3], dim=-1).cpu().numpy()* 255.)).save("feat_2.png")
    # save saliencies 
    Image.fromarray(np.uint8(sals[0, ..., 0].cpu().numpy()*255.)).save("sal.png")
    Image.fromarray(np.uint8(sals_1[0, ..., 0].cpu().numpy()*255.)).save("sal_1.png")
    Image.fromarray(np.uint8(sals_2[0, ..., 0].cpu().numpy()*255.)).save("sal_2.png")
    assert False, "don't forget to comment all debugging"
    assert False, "are your sure everything is right?" 
    '''




