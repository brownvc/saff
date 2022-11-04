import torch
import torchvision
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

@torch.no_grad()
def main():
    parser = config_parser()
    args = parser.parse_args()

    root_dir = "../data/dino_material"
    print("For now, stride=1, and no PCA")
    stride = 64

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
        while start_height < height:
            start_width = 0
            while True:
                windows.append((start_height, start_width))
                if start_width + dwidth > width:
                    break 
                start_width += stride
            if start_height + dheight > height:
                break
            start_height += stride
        windows_1 = []
        start_height = 0
        while start_height < dheight*2:
            start_width = 0
            while start_width < dwidth*2:
                windows_1.append((start_height, start_width))
                if start_width + dwidth > dwidth*2:
                    break 
                start_width += stride
            if start_height + dheight > dheight*2:
                break
            start_height += stride
        start_height = 0
        start_width = 0
        #assert False, [len(windows), len(windows_1)]
        
        padder = torch.nn.ReflectionPad2d((0, dwidth//2, 0, dheight//2))

        for split, imgs in zip(["training", "nv_spatial", "nv_static"], [training_imgs, nv_spatial_imgs, nv_static_imgs]):
            outdir = os.path.join(root_dir, scene, split)
            os.makedirs(outdir, exist_ok=True)

            

            tmp = {"feats": None,
                    "sals": None,
                    "feats_1": None,
                    "sals_1": None,
                    "feats_2": None,
                    "sals_2": None,
                    "batch": None,
                    "feat_raw": None,
                    "sal_raw": None,
                    "feats_raw": None,
                    "sals_raw": None
            }
            
            
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

                
                #pad img_1 and img_2
                img_1 = torch.from_numpy(img_1).permute(2, 0, 1)
                img_1 = padder(img_1).permute(1, 2, 0).numpy()
                #assert False, img_1.shape
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = padder(img).permute(1, 2, 0).numpy()
                #assert False, img_2.shape
                #Image.fromarray(np.uint8(img_1*255.)).save("img_1.png")
                #Image.fromarray(np.uint8(img_2*255.)).save("img_2.png")
                #assert False, [img_1.shape, img_2.shape, height, width, dheight, dwidth]
                

                #assert False, "add debugger for all affected by padding below"
                print(f"Rendering for split {split}, image {image_id}, level 2:")
                if tmp["feats_2"] is not None:
                    tmp["feats_2"] = tmp["feats_2"].to(device)
                    tmp["sals_2"] = tmp["sals_2"].to(device)
                    tmp["feats"] = tmp["feats"].cpu()
                    tmp["sals"] = tmp["sals"].cpu()
                with torch.no_grad():
                    tmp["batch"] = torch.from_numpy(img_2).permute(2, 0, 1)[None, ...].to(device)
                    tmp["feat_raw"] = extractor.extract_descriptors(tmp["batch"], args.layer, args.facet, args.bin)
                    tmp["feat_raw"] = tmp["feat_raw"].view(tmp["batch"].shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)        
                    tmp["feat_raw"] = F.interpolate(tmp["feat_raw"], size=(dheight, dwidth), mode='nearest')
                    if tmp["feats_2"] is None:
                        tmp["feats_2"] = torch.zeros((dheight, dwidth, tmp["feat_raw"].shape[1])).to(device)
                        tmp["sals_2"] = torch.zeros((dheight, dwidth, 1)).to(device)     
                        tmp["counter_2"] = torch.zeros((dheight, dwidth, 1))
                    
                    #assert False, [feats_2.shape, feat_raw.shape]
                    tmp["feats_2"] = tmp["feat_raw"].permute(0, 2, 3, 1)[0]
                    tmp["counter_2"] += 1
                    tmp["sal_raw"] = saliency_extractor.extract_saliency_maps(tmp["batch"])
                    tmp["sal_raw"] = tmp["sal_raw"].view(tmp["batch"].shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                    tmp["sal_raw"] = F.interpolate(tmp["sal_raw"], size=(dheight, dwidth), mode='nearest')
                    #print(sals_2.device, sal_raw.device)
                    tmp["sals_2"] = tmp["sal_raw"].permute(0, 2, 3, 1)[0]
                tmp["feats_2"] = tmp["feats_2"].cpu()
                tmp["sals_2"] = tmp["sals_2"].cpu()
                print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

                print(f"Rendering for split {split}, image {image_id}, level 1:")
                #start_height = 0
                if tmp["feats_1"] is not None:
                    tmp["feats_1"] = tmp["feats_1"].to(device)
                    tmp["sals_1"] = tmp["sals_1"].to(device)
                pbar = tqdm(total=len(windows_1))
                tmp["batch"] = None
                pos = []
                for start_height, start_width in windows_1:
                    if tmp["batch"] is None:
                        tmp["batch"] = torch.from_numpy(img_1[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]
                        #assert False, batch.shape
                    else:
                        tmp["batch"] = torch.cat([tmp["batch"], torch.from_numpy(img_1[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]], dim=0)
                        #assert False, batch.shape
                    pos.append((start_height, start_width))
                    if tmp["batch"].shape[0] >= dino_batch_size_1:
                        with torch.no_grad():
                            tmp["batch"] = tmp["batch"].to(device)
                            tmp["feats_raw"] = extractor.extract_descriptors(tmp["batch"], args.layer, args.facet, args.bin)
                            tmp["feats_raw"] = tmp["feats_raw"].view(tmp["batch"].shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            tmp["feats_raw"] = F.interpolate(tmp["feats_raw"], size=(dheight, dwidth), mode='nearest')
                            tmp["sals_raw"] = saliency_extractor.extract_saliency_maps(tmp["batch"])
                            tmp["sals_raw"] = tmp["sals_raw"].view(tmp["batch"].shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            tmp["sals_raw"] = F.interpolate(tmp["sals_raw"], size=(dheight, dwidth), mode='nearest')
                        if tmp["feats_1"] is None:
                            tmp["feats_1"] = torch.zeros((2*dheight, 2*dwidth, tmp["feats_raw"].shape[1])).to(device)
                            tmp["sals_1"] = torch.zeros((2*dheight, 2*dwidth, 1)).to(device)     
                            tmp["counter_1"] = torch.zeros((2*dheight, 2*dwidth, 1))
                        #assert False, [feats_1.shape, feats_raw.shape]
                        for t in range(len(tmp["batch"])):
                            tmp["feat_raw"] = tmp["feats_raw"][t:t+1]
                            tmp["sal_raw"] = tmp["sals_raw"][t:t+1]
                            hstart, wstart = pos[t]
                            hstart_real = max(0, hstart)
                            wstart_real = max(0, wstart)
                            hend_real = min(2*dheight, hstart+dheight)
                            wend_real = min(2*dwidth, wstart+dwidth)
                            hstart_crop = max(0, -hstart)
                            hend_crop = min(dheight, 2*dheight-hstart)
                            wstart_crop = max(0, -wstart)
                            wend_crop = min(dwidth, 2*dwidth-wstart)
                            tmp["feats_1"][hstart_real : hend_real, wstart_real : wend_real] +=\
                             tmp["feat_raw"].permute(0, 2, 3, 1)[0, hstart_crop:hend_crop, wstart_crop: wend_crop]
                            tmp["counter_1"][hstart_real : hend_real, wstart_real : wend_real] += 1   
                            tmp["sals_1"][hstart_real : hend_real, wstart_real : wend_real] +=\
                             tmp["sal_raw"].permute(0, 2, 3, 1)[0, hstart_crop:hend_crop, wstart_crop: wend_crop]
                        pbar.update(len(tmp["batch"]))
                        tmp["batch"] = None
                        pos = []
                        #break
                print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                tmp["feats_1"] = tmp["feats_1"].cpu()
                tmp["sals_1"] = tmp["sals_1"].cpu()
                    
                    
                pbar.close()

                

                

                print(f"Rendering for split {split}, image {image_id}, level 0:")
                #start_height = 0
                if tmp["feats"] is not None:
                    tmp["feats"] = tmp["feats"].to(device)
                    tmp["sals"] = tmp["sals"].to(device)
                pbar = tqdm(total=len(windows))
                tmp["batch"] = None
                pos = []
                #assert False, img.shape
                for start_height, start_width in windows:
                    #print(start_height, start_width, torch.from_numpy(img[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...].shape)
                    if tmp["batch"] is None:
                        tmp["batch"] = torch.from_numpy(img[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]
                        #assert False, batch.shape
                    else:
                        tmp["batch"] = torch.cat([tmp["batch"], torch.from_numpy(img[start_height:start_height+dheight, start_width:start_width+dwidth]).permute(2, 0, 1)[None, ...]], dim=0)
                        #assert False, batch.shape
                    pos.append((start_height, start_width))
                    if tmp["batch"].shape[0] >= dino_batch_size:
                        with torch.no_grad():
                            tmp["batch"] = tmp["batch"].to(device)
                            tmp["feats_raw"] = extractor.extract_descriptors(tmp["batch"], args.layer, args.facet, args.bin)
                            tmp["feats_raw"] = tmp["feats_raw"].view(tmp["batch"].shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            tmp["feats_raw"] = F.interpolate(tmp["feats_raw"], size=(dheight, dwidth), mode='nearest')
                            tmp["sals_raw"] = saliency_extractor.extract_saliency_maps(tmp["batch"])
                            tmp["sals_raw"] = tmp["sals_raw"].view(tmp["batch"].shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                            tmp["sals_raw"] = F.interpolate(tmp["sals_raw"], size=(dheight, dwidth), mode='nearest')
                            if tmp["feats"] is None:
                                tmp["feats"] = torch.zeros((height, width, tmp["feats_raw"].shape[1])).to(device)
                                tmp["sals"] = torch.zeros((height, width, 1)).to(device)     
                                tmp["counter"] = torch.zeros((height, width, 1))
                            #print(feats.shape, feats_raw.shape)
                            for t in range(len(tmp["batch"])):
                                tmp["feat_raw"] = tmp["feats_raw"][t:t+1]
                                tmp["sal_raw"] = tmp["sals_raw"][t:t+1]
                                hstart, wstart = pos[t]
                                #print(hstart+dheight, wstart+dwidth)
                                hstart_real = max(0, hstart)
                                wstart_real = max(0, wstart)
                                hend_real = min(height, hstart+dheight)
                                wend_real = min(width, wstart+dwidth)
                                hstart_crop = max(0, -hstart)
                                hend_crop = min(dheight, height-hstart)
                                wstart_crop = max(0, -wstart)
                                wend_crop = min(dwidth, width-wstart)
                                tmp["feats"][hstart_real : hend_real, wstart_real : wend_real] +=\
                                  tmp["feat_raw"].permute(0, 2, 3, 1)[0, hstart_crop:hend_crop, wstart_crop: wend_crop]
                                tmp["counter"][hstart_real : hend_real, wstart_real : wend_real] += 1   
                                tmp["sals"][hstart_real : hend_real, wstart_real : wend_real] +=\
                                  tmp["sal_raw"].permute(0, 2, 3, 1)[0, hstart_crop:hend_crop, wstart_crop: wend_crop]
                                #assert False
                                #tmp["feats"][image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += tmp["feat_raw"].permute(0, 2, 3, 1)
                                #tmp["counter"][image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += 1   
                                #tmp["sals"][image_id:image_id+1, hstart:hstart+dheight, wstart:wstart+dwidth] += tmp["sal_raw"].permute(0, 2, 3, 1)
                        pbar.update(len(tmp["batch"]))
                        tmp["batch"] = None
                        #break
                        pos = []
                        #torch.cuda.empty_cache()
                
                
                pbar.close()
                #break
                #print("Debuggin! break after first image")
                tmp["batch"] = None
                tmp["feat_raw"] = None
                tmp["sal_raw"] = None
                tmp["feats_raw"] = None
                tmp["sals_raw"] = None
            
                torch.cuda.empty_cache()
                print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                #break

                for item in tmp:
                    if tmp[item] is None:
                        continue
                    tmp[item] = tmp[item].cpu()
                torch.save(tmp["feats"], os.path.join(outdir, f"{image_id}_feats.pt"))
                torch.save(tmp["sals"], os.path.join(outdir, f"{image_id}_sals.pt"))
                torch.save(tmp["counter"], os.path.join(outdir, f"{image_id}_counter.pt"))

                torch.save(tmp["feats_1"], os.path.join(outdir, f"{image_id}_feats_1.pt"))
                torch.save(tmp["sals_1"], os.path.join(outdir, f"{image_id}_sals_1.pt"))
                torch.save(tmp["counter_1"], os.path.join(outdir, f"{image_id}_counter_1.pt"))
                
                torch.save(tmp["feats_2"], os.path.join(outdir, f"{image_id}_feats_2.pt"))
                torch.save(tmp["sals_2"], os.path.join(outdir, f"{image_id}_sals_2.pt"))
                torch.save(tmp["counter_2"], os.path.join(outdir, f"{image_id}_counter_2.pt"))
                
                
                #print("start debugging visualize dino")
                #dino = tmp["feats"] / (1e-16+tmp["counter"])
                #pca = PCA(n_components=3).fit(dino.contiguous().view(-1, dino.shape[-1])[::100])
                #old_shape = dino.shape
                #feats = torch.from_numpy(pca.transform(dino.contiguous().view(-1, dino.shape[-1]).numpy())).view(old_shape[0], old_shape[1], 3)
                #for comp_idx in range(3):
                #    comp = feats[..., comp_idx]
                #    comp_min = torch.min(comp)
                #    comp_max = torch.max(comp)
                #    comp = (comp - comp_min) / (comp_max - comp_min)
                #    feats[..., comp_idx] = comp
                
                #torchvision.transforms.functional.to_pil_image(feats.permute(2, 0, 1)).save("test.png")
                for item in tmp:
                    tmp[item] = None
                #assert False
                #break
            

            
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


if __name__ == "__main__":
    main()



