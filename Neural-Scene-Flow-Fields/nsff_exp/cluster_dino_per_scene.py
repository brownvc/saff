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
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    saliency_extractor = extractor
    

    os.makedirs(out_dir, exist_ok=True)
    feats = None
    sals = None
    H = None
    W = None
    num_samples_per_image = []
    #img_dirs = []
    ##tmp_idx = 0
    #while f'{tmp_idx}.png' in os.listdir(root_dir):
    #    img_dirs.append(f'{tmp_idx}.png')
    #   tmp_idx += 1
    #    #sorted(os.listdir(os.path.join(root_dir, name)))
    #    #assert False, img_dirs
    img_dirs = [img for img in os.listdir(root_dir)]
    img_dirs = sorted(img_dirs)
    num_img = len(img_dirs)
    #assert False, img_dirs
    for img in img_dirs:
        #print(img)
        if not img.endswith('.png'):
            continue
        if H is None:
            tmp = cv2.imread(os.path.join(root_dir, img))
            H = tmp.shape[0]
            W = tmp.shape[1]
            #assert False, [H, W]
        batch, _ = extractor.preprocess(os.path.join(root_dir, img), load_size)
        
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
        cv2.imwrite(os.path.join(out_dir, f"feat_{save_id}.png"), pca_feats[save_id] * 255.)
        cv2.imwrite(os.path.join(out_dir, f"sal_{save_id}.png"), sals.view(-1, H, W).cpu().numpy()[save_id] * 255.)
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
    faiss.write_index(algorithm.index, os.path.join(out_dir, "large.index")) 
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
                if c1 in label_mapper:
                    label_mapper[c2] = label_mapper[c1]
                else:
                    label_mapper[c2] = c1
                break    
    pickle.dump(label_mapper, open(os.path.join(out_dir, "label_mapper.pkl"), 'wb'))
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
    with open(os.path.join(out_dir, "salient.npy"), "wb") as f:
        np.save(f, salient_labels)        
    

    labels[~np.isin(labels, salient_labels)] = -1
    labels_per_image = np.split(labels, np.cumsum(num_samples_per_image))
    #assert False, labels_per_image[0].shape
    #os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    for idx, (image_labels_no_merge_no_salient, image_labels_no_salient, final_labels) in enumerate(zip(labels_per_image_no_merge_no_salient, labels_per_image_no_salient, labels_per_image)):
        #assert False, [image_labels_no_merge_no_salient.shape, final_labels.shape]
        #assert False, [type(final_labels), final_labels.shape]
        img_clu = d3_41_colors_rgb[np.resize(final_labels, (H, W))]
        #assert False, img_clu.shape
        #img_clu.reshape((H, W, 3))
        cv2.imwrite(os.path.join(out_dir, f"{idx}.png"), img_clu)

        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster sems')
    
    parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image sets.')
    parser.add_argument("--out_dir", type=str, required=True)
    #parser.add_argument("--wfeat_id", type=int, required=True)
    #parser.add_argument("--wsal_id", type=int, required=True)
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

    #root_dir = "../../data/test_data" 
    #out_dir = "../../data/dino_masks"  
    cluster_feats(args.root_dir, args.out_dir, 
        args.load_size, args.stride, args.model_type, args.facet, args.layer, args.bin, num_components=64)
    
    