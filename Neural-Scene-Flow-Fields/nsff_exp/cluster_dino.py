import sys
sys.path.append("../../dino_utils")
from cosegmentation import *
from pca import *
from pyramid import *
import torch
import os
import cv2
import faiss
from tqdm import tqdm
import pickle
from run_nerf_helpers import d3_41_colors_rgb
import imageio
import torch.nn.functional as F
import copy

from torchvision.utils import make_grid

def preprocess_feats(feats, sample_interval, skip_norm=False):
    all_descriptors = torch.cat(feats, dim=0).contiguous()
    normalized_all_descriptors = all_descriptors.float().cpu().numpy()
    #print(np.unique(normalized_all_descriptors))
    if not skip_norm:
        faiss.normalize_L2(normalized_all_descriptors)
    #print(np.unique(normalized_all_descriptors))
    sampled_descriptors_list = [x[::sample_interval, :] for x in feats]
    all_sampled_descriptors_list = torch.cat(sampled_descriptors_list, dim=0).contiguous()
    normalized_all_sampled_descriptors = all_sampled_descriptors_list.float().cpu().numpy()
    if not skip_norm:
        faiss.normalize_L2(normalized_all_sampled_descriptors)
    return normalized_all_descriptors, normalized_all_sampled_descriptors

@torch.no_grad()
def cluster_feats(root_dir, out_dir, load_size, stride, model_type, facet, layer, bin, num_components=64, sample_interval=5, n_cluster=25, elbow=0.975, similarity_thresh=0.5, thresh=0.07, votes_percentage=70):
    scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "Umbrella"]
    splits = ['train', 'nv_spatial', 'nv_static']

    device='cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    saliency_extractor = extractor
    
    num_img = 24

    # fit to get index, label_mapper and salient_labels
    for scene in scenes:
        #split = "train"
        name = scene + '_train'
        #print(os.path.join(root_dir, name))
        assert os.path.isdir(os.path.join(root_dir, name)), "no such directory"
        os.makedirs(os.path.join(out_dir, scene), exist_ok=True)
        feats = None
        sals = None
        H = None
        W = None
        num_samples_per_image = []
        img_dirs = []
        tmp_idx = 0
        while f'{tmp_idx}.jpg' in os.listdir(os.path.join(root_dir, name)):
            img_dirs.append(f'{tmp_idx}.jpg')
            tmp_idx += 1
            #sorted(os.listdir(os.path.join(root_dir, name)))
            #assert False, img_dirs
        for img in img_dirs:
            #print(img)
            if not img.endswith('.jpg'):
                continue
            if H is None:
                tmp = cv2.imread(os.path.join(root_dir, name, img))
                H = tmp.shape[0]
                W = tmp.shape[1]
                #assert False, [H, W]
            batch, _ = extractor.preprocess(os.path.join(root_dir, name, img), load_size)
           
            feat_raw = extractor.extract_descriptors(batch.to(device), layer, facet, bin)
            feat_raw = feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1)
            sal_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
            sal_raw = sal_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1)

            if feats is None:
                feats = feat_raw
                sals = sal_raw
            else:
                feats = torch.cat([feats, feat_raw], dim=0)
                sals = torch.cat([sals, sal_raw], dim=0)
            num_samples_per_image.append(H*W)
        feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None)
        old_shape = feats.shape
        feats = feats.view(-1, feats.shape[-1])
        pca = PCA(n_components=num_components).fit(feats.cpu())
        pca_feats = pca.transform(feats.cpu())
        feats = pca_feats.reshape((old_shape[0], old_shape[1], old_shape[2], -1))
        feats = torch.nn.functional.interpolate(torch.from_numpy(feats).permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1)
        pca_color = PCA(n_components=3).fit(feats.view(-1, feats.shape[-1]).cpu().numpy())
        #print("I am done")
        pca_feats = pca_color.transform(feats.view(-1, feats.shape[-1]).cpu().numpy())
        #print("I am done")
        pca_feats = pca_feats.reshape((-1, H, W, pca_feats.shape[-1]))
        for comp_idx in range(3):
            comp = pca_feats[..., comp_idx]
            comp_min = comp.min(axis=(0, 1))
            comp_max = comp.max(axis=(0, 1))
            comp_img = (comp - comp_min) / (comp_max - comp_min)
            pca_feats[..., comp_idx] = comp_img
        
        feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None).numpy()
        sals = torch.nn.functional.interpolate(sals.permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1).view(sals.shape[0], -1)

        
        for save_id in range(len(pca_feats)):
            cv2.imwrite(os.path.join(out_dir, scene, f"train_feat_{save_id}.png"), pca_feats[save_id] * 255.)
            cv2.imwrite(os.path.join(out_dir, scene, f"train_sal_{save_id}.png"), sals.view(-1, H, W).cpu().numpy()[save_id] * 255.)
        #assert False, "Pause and modify below"
        feature = feats.reshape((-1, num_components)).astype(np.float32)
        sampled_feature = np.ascontiguousarray(feature[::sample_interval])   
        sum_of_squared_dists = []
        n_cluster_range = list(range(1, n_cluster))
        for n_clu in tqdm(n_cluster_range):
            algorithm = faiss.Kmeans(d=feature.shape[-1], k=n_clu, gpu=False, niter=300, nredo=10, seed=1234, verbose=False)
            algorithm.train(sampled_feature)
            squared_distances, labels = algorithm.index.search(feature, 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / feature.shape[0])
            if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):    
                break
        faiss.write_index(algorithm.index, os.path.join(out_dir, scene, "large.index")) 
        num_labels = np.max(n_clu) + 1
        labels_per_image_no_merge_no_salient = np.split(labels, np.cumsum(num_samples_per_image))

        centroids = algorithm.centroids
        sims = -np.ones((len(centroids), len(centroids)))
        #assert samples["dinos"].shape[-1] == 64
        for c1 in range(len(centroids)):
            item_1 = centroids[c1][:64]
            for c2 in range(c1+1, len(centroids)):
                item_2 = centroids[c2][:64]
                sims[c1, c2] = np.dot(item_1, item_2) / (np.linalg.norm(item_1) * np.linalg.norm(item_2))
                print(c1, c2, sims[c1, c2])
        label_mapper = {}   
        for c2 in range(len(centroids)):
            for c1 in range(c2):
                if sims[c1, c2] > similarity_thresh:
                    label_mapper[c2] = c1
                    break    
        pickle.dump(label_mapper, open(os.path.join(out_dir, scene, "label_mapper.pkl"), 'wb'))
        for key in label_mapper:
            print(key, label_mapper[key])
        for c1 in range(len(centroids)):
            key = len(centroids) - c1 - 1
            if key in label_mapper:
                labels[labels == key] = label_mapper[key]
        labels_per_image_no_salient = np.split(labels, np.cumsum(num_samples_per_image))

        votes = np.zeros(num_labels)
        for image_labels, saliency_map in zip(labels_per_image_no_salient, sals):
            #assert False, [saliency_map.shape, (image_labels[:, 0] == 0).shape]
            for label in range(num_labels):
                label_saliency = saliency_map[image_labels[:, 0] == label].mean()
                if label_saliency > thresh:
                    votes[label] += 1
        print(votes)
        salient_labels = np.where(votes >= np.ceil(num_img * votes_percentage / 100))
        with open(os.path.join(out_dir, scene, "salient.npy"), "wb") as f:
            np.save(f, salient_labels)        
        

        labels[~np.isin(labels, salient_labels)] = -1
        labels_per_image = np.split(labels, np.cumsum(num_samples_per_image))
        #assert False, labels_per_image[0].shape
        os.makedirs(os.path.join(out_dir, scene, "train"), exist_ok=True)
        for idx, (image_labels_no_merge_no_salient, image_labels_no_salient, final_labels) in enumerate(zip(labels_per_image_no_merge_no_salient, labels_per_image_no_salient, labels_per_image)):
            #assert False, [image_labels_no_merge_no_salient.shape, final_labels.shape]
            #assert False, [type(final_labels), final_labels.shape]
            img_clu = d3_41_colors_rgb[np.resize(final_labels, (H, W))]
            #assert False, img_clu.shape
            #img_clu.reshape((H, W, 3))
            cv2.imwrite(os.path.join(out_dir, scene, "train", f"{idx}.png"), img_clu)

        for split in ["nv_spatial", "nv_static"]:
            #split = "train"
            name = scene + '_' + split
            #print(os.path.join(root_dir, name))
            assert os.path.isdir(os.path.join(root_dir, name)), "no such directory"
            os.makedirs(os.path.join(out_dir, scene), exist_ok=True)
            feats = None
            sals = None
            H = None
            W = None
            num_samples_per_image = []
            img_dirs = []
            tmp_idx = 0
            while f'{tmp_idx}.jpg' in os.listdir(os.path.join(root_dir, name)):
                img_dirs.append(f'{tmp_idx}.jpg')
                tmp_idx += 1
                #sorted(os.listdir(os.path.join(root_dir, name)))
                #assert False, img_dirs
            for img in img_dirs:
                #print(img)
                if not img.endswith('.jpg'):
                    continue
                if H is None:
                    tmp = cv2.imread(os.path.join(root_dir, name, img))
                    H = tmp.shape[0]
                    W = tmp.shape[1]
                    #assert False, [H, W]
                batch, _ = extractor.preprocess(os.path.join(root_dir, name, img), load_size)
            
                feat_raw = extractor.extract_descriptors(batch.to(device), layer, facet, bin)
                feat_raw = feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1)
                sal_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                sal_raw = sal_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1)

                if feats is None:
                    feats = feat_raw
                    sals = sal_raw
                else:
                    feats = torch.cat([feats, feat_raw], dim=0)
                    sals = torch.cat([sals, sal_raw], dim=0)
                num_samples_per_image.append(H*W)
            feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None)
            old_shape = feats.shape
            feats = feats.view(-1, feats.shape[-1])
            #pca = PCA(n_components=num_components).fit(feats.cpu())
            pca_feats = pca.transform(feats.cpu())
            feats = pca_feats.reshape((old_shape[0], old_shape[1], old_shape[2], -1))
            feats = torch.nn.functional.interpolate(torch.from_numpy(feats).permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1)
            
            #pca_color = PCA(n_components=3).fit(feats.view(-1, feats.shape[-1]).cpu().numpy())
            #print("I am done")
            pca_feats = pca_color.transform(feats.view(-1, feats.shape[-1]).cpu().numpy())
            #print("I am done")
            pca_feats = pca_feats.reshape((-1, H, W, pca_feats.shape[-1]))
            for comp_idx in range(3):
                comp = pca_feats[..., comp_idx]
                comp_min = comp.min(axis=(0, 1))
                comp_max = comp.max(axis=(0, 1))
                comp_img = (comp - comp_min) / (comp_max - comp_min)
                pca_feats[..., comp_idx] = comp_img
            
            feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None).numpy()
            sals = torch.nn.functional.interpolate(sals.permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1).view(sals.shape[0], -1)

            
            for save_id in range(len(pca_feats)):
                cv2.imwrite(os.path.join(out_dir, scene, f"{split}_feat_{save_id}.png"), pca_feats[save_id] * 255.)
                cv2.imwrite(os.path.join(out_dir, scene, f"{split}_sal_{save_id}.png"), sals.view(-1, H, W).cpu().numpy()[save_id] * 255.)
            #assert False, "Pause and modify below"
            
            

            feature = feats.reshape((-1, num_components)).astype(np.float32)
            _, labels = algorithm.index.search(feature, 1)

            for key in label_mapper:
                labels[labels == key] = label_mapper[key]
            
            labels[~np.isin(labels, salient_labels)] = -1
            
            labels_per_image = np.split(labels, np.cumsum(num_samples_per_image))
            os.makedirs(os.path.join(out_dir, scene, split), exist_ok=True)
            for idx, final_labels in enumerate(labels_per_image):
                #assert False, [image_labels_no_merge_no_salient.shape, final_labels.shape]
                #assert False, [type(final_labels), final_labels.shape]
                img_clu = d3_41_colors_rgb[np.resize(final_labels, (H, W))]
                #assert False, img_clu.shape
                #img_clu.reshape((H, W, 3))
                cv2.imwrite(os.path.join(out_dir, scene, split, f"{idx}.png"), img_clu)


@torch.no_grad()
def cluster_feats_multi(root_dir, out_dir, load_size, stride, model_type, facet, layer, bin, num_components=64, sample_interval=5, n_cluster=25, elbow=0.975, similarity_thresh=0.5, thresh=0.07, votes_percentage=70):
    scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "umbrella"]
    splits = ['train', 'nv_spatial', 'nv_static']

    device='cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    saliency_extractor = extractor
    
    num_img = 24
    N_samples = [0, 0.5, 1]

    # fit to get index, label_mapper and salient_labels
    for scene in scenes:
        #split = "train"
        name = scene + '_train'
        #print(os.path.join(root_dir, name))
        assert os.path.isdir(os.path.join(root_dir, name)), "no such directory"
        os.makedirs(os.path.join(out_dir, scene), exist_ok=True)
        
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
        
        img_dirs = []
        tmp_idx = 0
        while f'{tmp_idx}.jpg' in os.listdir(os.path.join(root_dir, name)):
            img_dirs.append(f'{tmp_idx}.jpg')
            tmp_idx += 1
            #sorted(os.listdir(os.path.join(root_dir, name)))
        #assert False, img_dirs
        images = [imread(os.path.join(root_dir, name, f))[...,:3]/255. for f in img_dirs if f.endswith('.jpg')]
        #assert False, images
        images = np.stack(images, 0)
        #assert False, imgs.shape
        H = images.shape[1]
        W = images.shape[2]
        #imgs = [imageio.imread()]
        start = 0

        coords = []
        height, width = images.shape[1], images.shape[2]
        if height < width:
            dwidth = int(load_size / float(height) * width)
            dheight = load_size
        else:
            dheight = int(load_size / float(width) * height)
            dwidth = load_size
        
        # for each image
        while start < images.shape[0]:
            #coords = []
            for N_sample in N_samples:
                if N_sample == 0:
                    coords.append([start, 0, 0, images[0].shape[0], images[0].shape[1]])
                else:
                    height_size = int(dheight // N_sample)
                    width_size = int(dwidth // N_sample)                        
                    height_step = height_size // 2
                    width_step = width_size // 2
                    start_height = 0
                    while start_height < height - height_size:
                        start_width = 0
                        while start_width < width - width_size:
                            coords.append([start, start_height, start_width, start_height + height_size, start_width + width_size])
                            if start_width == width-width_size -1:
                                break
                            start_width = min(width-width_size-1, start_width + width_step)
                        if start_height == height-height_size - 1:
                            break
                        start_height = min(height - height_size-1, start_height + height_step)
            start += 1
            #coordss.append(coords)
        #print(coords)
        #assert False, len(coords)
        start = None
        
        
        
        feats = None
        
        num_samples_per_image = [H*W] * images.shape[0]
        #assert False, num_samples_per_image
        #torch.zeros((len(images), images[0].shape[0], images[0].shape[1], args.n_components))
        counter = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], 1))     
        sals = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], 1))            
        for [image_id, start_height, start_width, end_height, end_width] in tqdm(coords):
            batch = images[image_id:image_id+1, start_height:end_height, start_width:end_width]
            batch = torch.tensor(batch).permute(0, 3, 1, 2).float()
            batch = F.interpolate(batch, size=(dheight, dwidth), mode='nearest')
            with torch.no_grad():
                feat_raw = extractor.extract_descriptors(batch.to(device), args.layer, args.facet, args.bin)
                feat_raw = feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                feat_raw = F.interpolate(feat_raw, size=(end_height - start_height, end_width - start_width), mode='nearest')
                if feats is None:
                    feats = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], feat_raw.shape[1]))
                #assert False, [feats.shape, feat_raw.shape]
                feats[image_id:image_id+1, start_height:end_height, start_width:end_width] += feat_raw.permute(0, 2, 3, 1).float().cpu()
                counter[image_id:image_id+1, start_height:end_height, start_width:end_width] += 1
                sal_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                sal_raw = sal_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                sal_raw = F.interpolate(sal_raw, size=(end_height - start_height, end_width - start_width), mode='nearest')
                sals[image_id:image_id+1, start_height:end_height, start_width:end_width] += sal_raw.permute(0, 2, 3, 1).cpu()
                
        feats /= 1e-16 + counter
        sals /= 1e-16 + counter
        
        feats = F.normalize(feats, p=2, eps=1e-12, dim=-1).cpu()
        counter = None
        sals = sals.cpu().view(sals.shape[0], -1, 1)
        #assert False, [feats.shape, sals.shape]

        #print("I am done")
        pca = PCA(n_components=num_components).fit(feats.view(-1, feats.shape[-1])[::100])
        #print("I am done")
        #print("I am done")
        split_idxs = np.array([images.shape[1] * images.shape[2] for _ in range(images.shape[0])])
        split_idxs = np.cumsum(split_idxs)
        feats = np.split(feats.view(-1, feats.shape[-1]).numpy(), split_idxs[:-1], axis=0)
        #print("I am done")
        feats = [pca.transform(feat) for feat in feats]
        feats = torch.from_numpy(np.concatenate(feats, axis=0)).view(images.shape[0], images.shape[1], images.shape[2], -1)
        #assert False, [len(num_patches_list), pca_feats.shape]
        
        pca_color = PCA(n_components=3).fit(feats.view(-1, feats.shape[-1]).cpu().numpy())
        #print("I am done")
        pca_feats = pca_color.transform(feats.view(-1, feats.shape[-1]).cpu().numpy())
        #print("I am done")
        pca_feats = pca_feats.reshape((-1, images[0].shape[0], images[0].shape[1], pca_feats.shape[-1]))
        for comp_idx in range(3):
            comp = pca_feats[..., comp_idx]
            comp_min = comp.min(axis=(0, 1))
            comp_max = comp.max(axis=(0, 1))
            comp_img = (comp - comp_min) / (comp_max - comp_min)
            pca_feats[..., comp_idx] = comp_img
        for save_id in range(len(pca_feats)):
            cv2.imwrite(os.path.join(out_dir, scene, f"train_feat_{save_id}.png"), pca_feats[save_id] * 255.)
            cv2.imwrite(os.path.join(out_dir, scene, f"train_sal_{save_id}.png"), sals.view(-1, images.shape[1], images.shape[2]).numpy()[save_id] * 255.)
        #assert False
        
        feature = F.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None).view(-1, num_components).numpy().astype(np.float32)        
        sampled_feature = np.ascontiguousarray(feature[::sample_interval])   
        sum_of_squared_dists = []
        n_cluster_range = list(range(1, n_cluster))
        for n_clu in tqdm(n_cluster_range):
            algorithm = faiss.Kmeans(d=feature.shape[-1], k=n_clu, gpu=False, niter=300, nredo=10, seed=1234, verbose=False)
            algorithm.train(sampled_feature)
            squared_distances, labels = algorithm.index.search(feature, 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / feature.shape[0])
            if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):    
                break
        faiss.write_index(algorithm.index, os.path.join(out_dir, scene, "large.index")) 
        num_labels = np.max(n_clu) + 1
        labels_per_image_no_merge_no_salient = np.split(labels, np.cumsum(num_samples_per_image))

        centroids = algorithm.centroids
        sims = -np.ones((len(centroids), len(centroids)))
        #assert samples["dinos"].shape[-1] == 64
        for c1 in range(len(centroids)):
            item_1 = centroids[c1][:64]
            for c2 in range(c1+1, len(centroids)):
                item_2 = centroids[c2][:64]
                sims[c1, c2] = np.dot(item_1, item_2) / (np.linalg.norm(item_1) * np.linalg.norm(item_2))
                print(c1, c2, sims[c1, c2])
        label_mapper = {}   
        for c2 in range(len(centroids)):
            for c1 in range(c2):
                if sims[c1, c2] > similarity_thresh:
                    label_mapper[c2] = c1
                    break    
        pickle.dump(label_mapper, open(os.path.join(out_dir, scene, "label_mapper.pkl"), 'wb'))
        for key in label_mapper:
            print(key, label_mapper[key])
        for c1 in range(len(centroids)):
            key = len(centroids) - c1 - 1
            if key in label_mapper:
                labels[labels == key] = label_mapper[key]
        labels_per_image_no_salient = np.split(labels, np.cumsum(num_samples_per_image))

        votes = np.zeros(num_labels)
        for image_labels, saliency_map in zip(labels_per_image_no_salient, sals):
            #assert False, [saliency_map.shape, (image_labels[:, 0] == 0).shape]
            for label in range(num_labels):
                label_saliency = saliency_map[image_labels[:, 0] == label].mean()
                if label_saliency > thresh:
                    votes[label] += 1
        print(votes)
        salient_labels = np.where(votes >= np.ceil(num_img * votes_percentage / 100))
        with open(os.path.join(out_dir, scene, "salient.npy"), "wb") as f:
            np.save(f, salient_labels)        
        

        labels[~np.isin(labels, salient_labels)] = -1
        labels_per_image = np.split(labels, np.cumsum(num_samples_per_image))
        #assert False, labels_per_image[0].shape
        os.makedirs(os.path.join(out_dir, scene, "train"), exist_ok=True)
        for idx, (image_labels_no_merge_no_salient, image_labels_no_salient, final_labels) in enumerate(zip(labels_per_image_no_merge_no_salient, labels_per_image_no_salient, labels_per_image)):
            #assert False, [image_labels_no_merge_no_salient.shape, final_labels.shape]
            #assert False, [type(final_labels), final_labels.shape]
            img_clu = d3_41_colors_rgb[np.resize(final_labels, (H, W))]
            #assert False, img_clu.shape
            #img_clu.reshape((H, W, 3))
            cv2.imwrite(os.path.join(out_dir, scene, "train", f"{idx}.png"), img_clu)

        for split in ["nv_spatial", "nv_static"]:
            #split = "train"
            name = scene + '_' + split
            #print(os.path.join(root_dir, name))
            assert os.path.isdir(os.path.join(root_dir, name)), "no such directory"
            os.makedirs(os.path.join(out_dir, scene), exist_ok=True)
            
            def imread(f):
                if f.endswith('png'):
                    return imageio.imread(f, ignoregamma=True)
                else:
                    return imageio.imread(f)
            img_dirs = []
            tmp_idx = 0
            while f'{tmp_idx}.jpg' in os.listdir(os.path.join(root_dir, name)):
                img_dirs.append(f'{tmp_idx}.jpg')
                tmp_idx += 1
                #sorted(os.listdir(os.path.join(root_dir, name)))
            #assert False, img_dirs
            images = [imread(os.path.join(root_dir, name, f))[...,:3]/255. for f in img_dirs if f.endswith('.jpg')]
            #assert False, [sorted(os.listdir(os.path.join(root_dir, name))), os.listdir(os.path.join(root_dir, name))]
            #images = [imread(os.path.join(root_dir, name, f))[...,:3]/255. for f in sorted(os.listdir(os.path.join(root_dir, name))) if f.endswith('.jpg')]
            images = np.stack(images, 0)
            #assert False, imgs.shape
            H = images.shape[1]
            W = images.shape[2]
            #imgs = [imageio.imread()]
            start = 0

            coords = []
            height, width = images.shape[1], images.shape[2]
            if height < width:
                dwidth = int(load_size / float(height) * width)
                dheight = load_size
            else:
                dheight = int(load_size / float(width) * height)
                dwidth = load_size
            
            # for each image
            while start < images.shape[0]:
                #coords = []
                for N_sample in N_samples:
                    if N_sample == 0:
                        coords.append([start, 0, 0, images[0].shape[0], images[0].shape[1]])
                    else:
                        height_size = int(dheight // N_sample)
                        width_size = int(dwidth // N_sample)                        
                        height_step = height_size // 2
                        width_step = width_size // 2
                        start_height = 0
                        while start_height < height - height_size:
                            start_width = 0
                            while start_width < width - width_size:
                                coords.append([start, start_height, start_width, start_height + height_size, start_width + width_size])
                                if start_width == width-width_size -1:
                                    break
                                start_width = min(width-width_size-1, start_width + width_step)
                            if start_height == height-height_size - 1:
                                break
                            start_height = min(height - height_size-1, start_height + height_step)
                start += 1
                #coordss.append(coords)
            #print(coords)
            #assert False, len(coords)
            start = None
            
            
            
            feats = None
            
            num_samples_per_image = [H*W] * images.shape[0]
            #assert False, num_samples_per_image
            #torch.zeros((len(images), images[0].shape[0], images[0].shape[1], args.n_components))
            counter = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], 1))     
            sals = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], 1))            
            for [image_id, start_height, start_width, end_height, end_width] in tqdm(coords):
                batch = images[image_id:image_id+1, start_height:end_height, start_width:end_width]
                batch = torch.tensor(batch).permute(0, 3, 1, 2).float()
                batch = F.interpolate(batch, size=(dheight, dwidth), mode='nearest')
                with torch.no_grad():
                    feat_raw = extractor.extract_descriptors(batch.to(device), args.layer, args.facet, args.bin)
                    feat_raw = feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                    feat_raw = F.interpolate(feat_raw, size=(end_height - start_height, end_width - start_width), mode='nearest')
                    if feats is None:
                        feats = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], feat_raw.shape[1]))
                    #assert False, [feats.shape, feat_raw.shape]
                    feats[image_id:image_id+1, start_height:end_height, start_width:end_width] += feat_raw.permute(0, 2, 3, 1).float().cpu()
                    counter[image_id:image_id+1, start_height:end_height, start_width:end_width] += 1
                    sal_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                    sal_raw = sal_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                    sal_raw = F.interpolate(sal_raw, size=(end_height - start_height, end_width - start_width), mode='nearest')
                    sals[image_id:image_id+1, start_height:end_height, start_width:end_width] += sal_raw.permute(0, 2, 3, 1).cpu()
                    
            feats /= 1e-16 + counter
            sals /= 1e-16 + counter
            
            feats = F.normalize(feats, p=2, eps=1e-12, dim=-1).cpu()
            counter = None
            sals = sals.cpu().view(sals.shape[0], -1, 1)
            #assert False, [feats.shape, sals.shape]

            #print("I am done")
            #pca = PCA(n_components=num_components).fit(feats.view(-1, feats.shape[-1])[::100])
            #print("I am done")
            #print("I am done")
            split_idxs = np.array([images.shape[1] * images.shape[2] for _ in range(images.shape[0])])
            split_idxs = np.cumsum(split_idxs)
            feats = np.split(feats.view(-1, feats.shape[-1]).numpy(), split_idxs[:-1], axis=0)
            #print("I am done")
            feats = [pca.transform(feat) for feat in feats]
            feats = torch.from_numpy(np.concatenate(feats, axis=0)).view(images.shape[0], images.shape[1], images.shape[2], -1)
            
            pca_feats = pca_color.transform(feats.view(-1, feats.shape[-1]).cpu().numpy())
            #print("I am done")
            pca_feats = pca_feats.reshape((-1, images[0].shape[0], images[0].shape[1], pca_feats.shape[-1]))
            for comp_idx in range(3):
                comp = pca_feats[..., comp_idx]
                comp_min = comp.min(axis=(0, 1))
                comp_max = comp.max(axis=(0, 1))
                comp_img = (comp - comp_min) / (comp_max - comp_min)
                pca_feats[..., comp_idx] = comp_img
            for save_id in range(len(pca_feats)):
                cv2.imwrite(os.path.join(out_dir, scene, f"{split}_feat_{save_id}.png"), pca_feats[save_id] * 255.)
                cv2.imwrite(os.path.join(out_dir, scene, f"{split}_sal_{save_id}.png"), sals.view(-1, images.shape[1], images.shape[2]).numpy()[save_id] * 255.)
            
            #assert False, [len(num_patches_list), pca_feats.shape]
            feature = F.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None).view(-1, num_components).numpy().astype(np.float32)
            #feature = feats.reshape((-1, num_components)).astype(np.float32)
            _, labels = algorithm.index.search(feature, 1)

            for key in label_mapper:
                labels[labels == key] = label_mapper[key]
            
            labels[~np.isin(labels, salient_labels)] = -1
            
            labels_per_image = np.split(labels, np.cumsum(num_samples_per_image))
            os.makedirs(os.path.join(out_dir, scene, split), exist_ok=True)
            for idx, final_labels in enumerate(labels_per_image):
                #assert False, [image_labels_no_merge_no_salient.shape, final_labels.shape]
                #assert False, [type(final_labels), final_labels.shape]
                img_clu = d3_41_colors_rgb[np.resize(final_labels, (H, W))]
                #assert False, img_clu.shape
                #img_clu.reshape((H, W, 3))
                cv2.imwrite(os.path.join(out_dir, scene, split, f"{idx}.png"), img_clu)


def cluster_feats_pyramid(root_dir, expname, wfeats, wsals, n_components, sample_interval=5, n_cluster=25, elbow=0.975, similarity_thresh=0.5, thresh=0.07, votes_percentage=70):
    
    scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "Umbrella"]
    os.makedirs(os.path.join(root_dir, expname), exist_ok=True)
    for scene in scenes:
        os.makedirs(os.path.join(root_dir, expname, scene), exist_ok=True)
        weights = [wfeats["wfeat"], wfeats["wfeat_1"], wfeats["wfeat_2"]]
        sal_weights = [wsals["wsal"], wsals["wsal_1"], wsals["wsal_2"]]
        feature, sals, pca = load_feat_sal(os.path.join(root_dir, scene, "training"), n_components, weights, sal_weights)
        #assert False, ret["featss"].shape
        #for item in ret:
        #    print(item, 1ret[item].shape)
        #assert False
        #assert False, [wfeats, wsals]
        #feats = torch.nn.functional.interpolate(torch.from_numpy(feats).permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1)
        #feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None).numpy()
        #sals = torch.nn.functional.interpolate(sals.permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1).view(sals.shape[0], -1)
        num_img = feature.shape[0]
        H, W = feature.shape[1], feature.shape[2]
        num_samples_per_image = [H*W]*num_img
        #feature = 0
        #sals = 0 
        #for level in ["", "_1", "_2"]:
        #    feature += wfeats[f"wfeat{level}"] * ret[f"feats{level}"]
        #    sals += wsals[f"wsal{level}"] * ret[f"sals{level}"]
        '''
        old_shape= feature.shape
        featss = feature
        cpca = PCA(n_components=3).fit(featss.view(-1, featss.shape[-1]))
        featss = torch.from_numpy(cpca.transform(featss.view(-1, featss.shape[-1]).numpy())).view(old_shape[0], old_shape[1], old_shape[2], -1)
        for comp_idx in range(3):
            comp = featss[..., comp_idx]
            comp_min = torch.min(comp)
            comp_max = torch.max(comp)
            comp = (comp - comp_min) / (comp_max - comp_min)
            featss[..., comp_idx] = comp
        time_idx = 0
        #salss = sals
        grids = None
        while os.path.exists(os.path.join(root_dir, scene, "training", f"{time_idx}_feats.pt")):
            feats = featss[time_idx]
            #sals = salss[time_idx]    
            grid = make_grid([feats.permute(2, 0, 1)*255.])
            #assert False, [feat.shape, sal.shape, grid.shape]
            #assert False, grid.shape
            if grids is None:
                grids = grid.permute(1, 2, 0)[None, ...]
            else:
                grids = torch.cat([grids, grid.permute(1,2,0)[None, ...]]) 
            time_idx += 1
            print(time_idx)
        if grids.shape[-2] % 2 == 1:
            grids = grids[:, :, :-1, :]
        torchvision.io.write_video("test.mp4", 
        grids, fps=1)
        print(f"Done with {root_dir}")
        
        assert False, [feature.shape, sals.shape]
        '''
        sals = sals.view(sals.shape[0], -1, 1)
        feature = torch.nn.functional.normalize(feature.view((-1, n_components)), dim=-1).numpy().astype(np.float32)
        sampled_feature = np.ascontiguousarray(feature[::sample_interval])   
        sum_of_squared_dists = []
        n_cluster_range = list(range(1, n_cluster))
        for n_clu in tqdm(n_cluster_range):
            algorithm = faiss.Kmeans(d=feature.shape[-1], k=n_clu, niter=300, nredo=10, seed=1234, verbose=False)
            algorithm.train(sampled_feature)
            squared_distances, labels = algorithm.index.search(feature, 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / feature.shape[0])
            if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):    
                break
        faiss.write_index(algorithm.index, os.path.join(root_dir, expname, scene, "large.index")) 
        num_labels = np.max(n_clu) + 1
        labels_per_image_no_merge_no_salient = copy.deepcopy(np.split(labels, np.cumsum(num_samples_per_image)))

        centroids = algorithm.centroids
        sims = -np.ones((len(centroids), len(centroids)))
        #assert samples["dinos"].shape[-1] == 64
        for c1 in range(len(centroids)):
            item_1 = centroids[c1][:64]
            for c2 in range(c1+1, len(centroids)):
                item_2 = centroids[c2][:64]
                sims[c1, c2] = np.dot(item_1, item_2) / (np.linalg.norm(item_1) * np.linalg.norm(item_2))
                print(c1, c2, sims[c1, c2])
        label_mapper = {}   
        for c2 in range(len(centroids)):
            for c1 in range(c2):
                if sims[c1, c2] > similarity_thresh:
                    label_mapper[c2] = c1
                    break    
        pickle.dump(label_mapper, open(os.path.join(root_dir, expname, scene, "label_mapper.pkl"), 'wb'))
        for key in label_mapper:
            print(key, label_mapper[key])
        for c1 in range(len(centroids)):
            key = len(centroids) - c1 - 1
            if key in label_mapper:
                labels[labels == key] = label_mapper[key]
        labels_per_image_no_salient = np.split(labels, np.cumsum(num_samples_per_image))

        votes = np.zeros(num_labels)
        for image_labels, saliency_map in zip(labels_per_image_no_salient, sals):
            #assert False, [saliency_map.shape, (image_labels[:, 0] == 0).shape]
            for label in range(num_labels):
                label_saliency = saliency_map[image_labels[:, 0] == label].mean()
                if label_saliency > thresh:
                    votes[label] += 1
        print(votes)
        salient_labels = np.where(votes >= np.ceil(num_img * votes_percentage / 100))
        with open(os.path.join(root_dir, expname, scene, "salient.npy"), "wb") as f:
            np.save(f, salient_labels)        
        

        labels[~np.isin(labels, salient_labels)] = -1
        labels_per_image = np.split(labels, np.cumsum(num_samples_per_image))
        #assert False, labels_per_image[0].shape
        os.makedirs(os.path.join(root_dir, expname, scene, "train"), exist_ok=True)
        for idx, (image_labels_no_merge_no_salient, image_labels_no_salient, final_labels) in enumerate(zip(labels_per_image_no_merge_no_salient, labels_per_image_no_salient, labels_per_image)):
            #assert False, [image_labels_no_merge_no_salient.shape, final_labels.shape]
            #assert False, [type(final_labels), final_labels.shape]
            img_clu = d3_41_colors_rgb[np.resize(image_labels_no_merge_no_salient, (H, W))]
            #assert False, img_clu.shape
            #img_clu.reshape((H, W, 3))
            cv2.imwrite(os.path.join(root_dir, expname, scene, "train", f"{idx}_raw.png"), img_clu)
            img_clu = d3_41_colors_rgb[np.resize(final_labels, (H, W))]
            #assert False, img_clu.shape
            #img_clu.reshape((H, W, 3))
            cv2.imwrite(os.path.join(root_dir, expname, scene, "train", f"{idx}.png"), img_clu)
        
        for split in ["nv_spatial", "nv_static"]:
            weights = [wfeats["wfeat"], wfeats["wfeat_1"], wfeats["wfeat_2"]]
            sal_weights = [wsals["wsal"], wsals["wsal_1"], wsals["wsal_2"]]
            feature, sals, _ = load_feat_sal(os.path.join(root_dir, scene, split), n_components, weights, sal_weights, pca=pca)
            #assert False, ret["featss"].shape
            #for item in ret:
            #    print(item, 1ret[item].shape)
            #assert False
            #assert False, [wfeats, wsals]
            #feats = torch.nn.functional.interpolate(torch.from_numpy(feats).permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1)
            #feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None).numpy()
            #sals = torch.nn.functional.interpolate(sals.permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1).view(sals.shape[0], -1)
            num_img = feature.shape[0]
            H, W = feature.shape[1], feature.shape[2]
            num_samples_per_image = [H*W]*num_img
            #ret = load_feat_sal(os.path.join(root_dir, scene, split), pcas)
            #for item in ret:
            #    print(item, ret[item].shape)
            #assert False
            #assert False, [wfeats, wsals]
            #feats = torch.nn.functional.interpolate(torch.from_numpy(feats).permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1)
            #feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None).numpy()
            #sals = torch.nn.functional.interpolate(sals.permute(0, 3, 1, 2), (H, W), mode="nearest").permute(0, 2, 3, 1).view(sals.shape[0], -1)
            #num_img = ret["feats"].shape[0]
            #H, W = ret["feats"].shape[1], ret["feats"].shape[2]
            #num_samples_per_image = [ret["feats"].shape[1] * ret["feats"].shape[2]]*num_img
            #feature = 0
            #sals = 0 
            #for level in ["", "_1", "_2"]:
            #    feature += wfeats[f"wfeat{level}"] * ret[f"feats{level}"]
            #    sals += wsals[f"wsal{level}"] * ret[f"sals{level}"]
            sals = sals.view(sals.shape[0], -1, 1)
            feature = torch.nn.functional.normalize(feature.view((-1, n_components)), dim=-1).numpy().astype(np.float32)
            
            _, labels = algorithm.index.search(feature, 1)

            for key in label_mapper:
                labels[labels == key] = label_mapper[key]
            
            labels[~np.isin(labels, salient_labels)] = -1
            
            labels_per_image = np.split(labels, np.cumsum(num_samples_per_image))
            os.makedirs(os.path.join(root_dir, expname, scene, split), exist_ok=True)
            for idx, final_labels in enumerate(labels_per_image):
                #assert False, [image_labels_no_merge_no_salient.shape, final_labels.shape]
                #assert False, [type(final_labels), final_labels.shape]
                img_clu = d3_41_colors_rgb[np.resize(final_labels, (H, W))]
                #assert False, img_clu.shape
                #img_clu.reshape((H, W, 3))
                cv2.imwrite(os.path.join(root_dir, expname, scene, split, f"{idx}.png"), img_clu)
        
        #assert False, "Pause"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster sems')
    
    parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image sets.')
    parser.add_argument("--wfeat_id", type=int, required=True)
    parser.add_argument("--wsal_id", type=int, required=True)
    '''
    parser.add_argument('--max_cluster', type=int, required=True, help='how many clusters')
    parser.add_argument('--depth_ratio', type=float, default=0, help="how much depth information to use")
    parser.add_argument('--pixel_ratio', type=float, default=0, help="how much pixel information to use")
    parser.add_argument('--pts_ratio', type=float, default=0, help="how much 3D points information to use")
    parser.add_argument('--use_gt_dino', action="store_true", help="whether use gt dino feature without reconstruction")
    parser.add_argument('--use_gt_sal', action="store_true", help="whether use gt saliency feature without reconstruction")
    parser.add_argument('--votes_percentage', default=75, type=int, help="percentage of votes needed for a cluster to "
    "be considered salient.")
    parser.add_argument('--thresh', default=0.065, type=float, help='saliency maps threshold to distinguish fg / bg.') 
    '''

    parser.add_argument('--load_size', default=128, type=int, help='load size of the input images. If None maintains'
                                                                    'original image size, if int resizes each image'
                                                                    'such that the smaller side is this number.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                                 small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                           Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                           vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--remove_outliers', default='False', type=str2bool, help="Remove outliers using cls token.")
    parser.add_argument('--load_algo', default='', type=str, help="load a trained kmeans or not")

    args = parser.parse_args()

    root_dir = "../../data/test_data" 
    out_dir = "../../data/dino_masks"  
    cluster_feats(root_dir, out_dir, args.load_size, args.stride, args.model_type, args.facet, args.layer, args.bin, num_components=64)
    
    assert False

    root_dir = "../../data/test_data" 
    out_dir = "../../data/dino_masks_multi"  
    cluster_feats_multi(root_dir, out_dir, args.load_size, args.stride, args.model_type, args.facet, args.layer, args.bin, num_components=64)
    assert False
    '''
    feats = load_feats(args.root_dir, sample_interval=100, max_cluster=args.max_cluster, elbow=0.975, use_gt_dino=args.use_gt_dino, use_gt_sal=args.use_gt_sal, depth_ratio=args.depth_ratio, pixel_ratio=args.pixel_ratio,
        pts_ratio=args.pts_ratio,
        load_size=args.load_size, stride=args.stride, model_type=args.model_type, facet=args.facet, layer=args.layer, bin=args.bin, remove_outliers=args.remove_outliers,
        votes_percentage=args.votes_percentage, thresh=args.thresh,
        load_algo=args.load_algo)
    '''


    wfeats_list = [
        {
            "wfeat": 1.,
            "wfeat_1": 0.,
            "wfeat_2": 0.
        },
        {
            "wfeat": 1/3.,
            "wfeat_1": 1/3.,
            "wfeat_2": 1/3.
        },
        {
            "wfeat": 1/7.,
            "wfeat_1": 2/7.,
            "wfeat_2": 4/7.
        },
        {
            "wfeat": 0.,
            "wfeat_1": 0.,
            "wfeat_2": 1.
        },
    ]
    wsals_list = [
        {
            "wsal": 1.,
            "wsal_1": 0.,
            "wsal_2": 0.
        },
        {
            "wsal": 1/3.,
            "wsal_1": 1/3.,
            "wsal_2": 1/3.
        },
        {
            "wsal": 1/7.,
            "wsal_1": 2/7.,
            "wsal_2": 4/7.
        },
        {
            "wsal": 0.,
            "wsal_1": 0.,
            "wsal_2": 1.
        },
    ]

    assert args.wfeat_id < len(wfeats_list), "not a valid feat weight id"
    assert args.wsal_id < len(wsals_list), "not a valid sal weight id"


    cluster_feats_pyramid(args.root_dir, f"cluster_dino_{args.wfeat_id}_{args.wsal_id}", wfeats_list[args.wfeat_id], wsals_list[args.wsal_id], n_components=64)