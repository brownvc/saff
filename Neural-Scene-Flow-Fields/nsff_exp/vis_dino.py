import sys
sys.path.append("../../dino_utils")
from cosegmentation import *
from pca import *
import torch
import os
import cv2

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
def load_feats(root_dir, sample_interval, max_cluster, load_size, stride, model_type, facet, layer, bin, elbow, remove_outliers, votes_percentage, thresh, save_png=True, depth_ratio=10, pixel_ratio=10, pts_ratio=10, use_gt_dino=False,
            use_gt_sal=False,
            prep_dino=True):
    feat_files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')])
    feats = []
    num_patches_list = []
    depths = []
    pixels = []
    pts = []
    use_depth = bool(depth_ratio > 0)
    use_pixel = bool(pixel_ratio > 0)
    use_pts = bool(pts_ratio > 0)
    saliency_maps_list = []
    if use_gt_dino or use_gt_sal:
        extractor = ViTExtractor(model_type, stride, device='cuda' if torch.cuda.is_available() else 'cpu')
        if use_gt_sal:
            saliency_extractor = extractor
    for i, feat_file in enumerate(feat_files):
        if i >= 35:
            break
        try:
            feat = torch.load(feat_file)[0]
            num_patches_list.append((feat.shape[-2], feat.shape[-1]))
            if use_gt_dino or use_gt_sal:
                batch, _ = extractor.preprocess(feat_file.replace('dinos', 'images').replace('.pt', '.jpg'), load_size)
                #batch = F.interpolate(batch, size=num_patches_list[-1], mode='nearest')
                #print(batch.shape)
            if use_gt_dino:
                descs = extractor.extract_descriptors(batch.to(feat.device), layer, facet, bin)
                #print(descs.shape)
                descs = descs.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1)
                if prep_dino:                
                    descs = torch.nn.functional.normalize(descs, p=2.0, dim=-1, eps=1e-12, out=None)
                    old_shape = descs.shape
                    descs = descs.view(-1, descs.shape[-1])
                    #assert False, feat.shape
                    pca = PCA(n_components=feat.shape[0]).fit(descs.cpu())
                    pca_descs = pca.transform(descs.cpu())
                    descs = torch.from_numpy(pca_descs).view(old_shape[0], old_shape[1], old_shape[2], feat.shape[0]).to(feat.device)
                    #assert False, [pca_feats.shape, old_shape]
                    #assert False, [len(num_patches_list), pca_feats.shape]
                    #split_idxs = np.array([num_patches[0] * num_patches[1] for num_patches in num_patches_list])
                    #split_idxs = np.cumsum(split_idxs)
                    #pca_per_image = np.split(pca_feats, split_idxs[:-1], axis=0)
                    #feats = torch.from_numpy(np.stack(pca_per_image, axis=0)).view(old_shape[0], old_shape[1], old_shape[2], -1)
                
                feat = torch.nn.functional.interpolate(descs.permute(0, 3, 1, 2), size=num_patches_list[-1], mode="bilinear")[0]
                #batch = F.interpolate(batch, size=(dheight, dwidth), mode='nearest')
                #assert False, [feat.shape, descs.shape]
            feats.append(feat.view(feat.shape[0], -1).permute(1, 0))
            if use_gt_sal:
                saliency_map = saliency_extractor.extract_saliency_maps(batch.to(feat.device))
                saliency_map = saliency_map.view(batch.shape[0], saliency_extractor.num_patches[0], saliency_extractor.num_patches[1], -1)
                saliency_map = torch.nn.functional.interpolate(saliency_map.permute(0, 3, 1, 2), size=num_patches_list[-1], mode="nearest")[0]
                saliency_maps_list.append(saliency_map[0].view(-1).cpu().numpy())
                #assert False, saliency_map.shape
                
            if use_depth or use_pts:
                depth = cv2.imread(feat_file.replace('dinos', 'depths').replace('.pt', '.jpg'), cv2.IMREAD_GRAYSCALE)/255.
            if use_depth:
                #print(feat_file.replace('dinos', 'depths').replace('.pt', '.jpg'))
                depths.append(torch.from_numpy(depth).view(-1, 1))
                #print(depth.shape)
            if use_pixel:
                W = feat.shape[-1]
                H = feat.shape[-2]
                xx, yy = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
                xx = (xx.t() - W/2.)/W
                yy = (yy.t() - H/2.)/H
                pts_2d_original = torch.stack([yy, xx], -1)#notice: different from in render_utils!
                pixels.append(pts_2d_original.view(-1, 2))
                #assert False, [feats[-1].shape, feat.shape, pts_2d_original.shape, pts_2d_original[-1, 0]]
            if use_pts:
                rays = torch.load(feat_file.replace('dinos', 'poses'))
                rays_o = rays[0]
                rays_d = rays[1]
                points = rays_o[...,None,:] + rays_d[...,None,:] * torch.from_numpy(depth[..., None]).to(rays_o.device)[...,:,None]
                #assert False, [torch.max(points[..., 0]), torch.min(points[..., 0]),
                #torch.max(points[..., 1]), torch.min(points[..., 1]),
                #torch.max(points[..., 2]), torch.min(points[..., 2]),]
                pts.append(points.view(-1, 3))

        except Exception as e: 
            print(e)
            break
    #assert False, pts[-1].shape
    #for depth in depths:
    #    print(torch.unique(depth))
    #assert False
    num_images = len(feats)
    normalized_all_descriptors, normalized_all_sampled_descriptors = \
    preprocess_feats(feats, sample_interval)
    if use_depth:
        normalized_all_depths, normalized_all_sampled_depths = \
        preprocess_feats(depths, sample_interval, skip_norm=True)
        normalized_all_descriptors = np.concatenate([normalized_all_descriptors, normalized_all_depths * depth_ratio], axis=-1)
        normalized_all_sampled_descriptors = np.concatenate([normalized_all_sampled_descriptors, normalized_all_sampled_depths * depth_ratio], axis=-1)
    if use_pixel:
        normalized_all_pixels, normalized_all_sampled_pixels = \
        preprocess_feats(pixels, sample_interval, skip_norm=True)
        normalized_all_descriptors = np.concatenate([normalized_all_descriptors, normalized_all_pixels * pixel_ratio], axis=-1)
        normalized_all_sampled_descriptors = np.concatenate([normalized_all_sampled_descriptors, normalized_all_sampled_pixels * pixel_ratio], axis=-1)
    if use_pts:
        normalized_all_pts, normalized_all_sampled_pts = \
        preprocess_feats(pts, sample_interval)
        normalized_all_descriptors = np.concatenate([normalized_all_descriptors, normalized_all_pts * pts_ratio], axis=-1)
        normalized_all_sampled_descriptors = np.concatenate([normalized_all_sampled_descriptors, normalized_all_sampled_pts * pts_ratio], axis=-1)
    
    #assert False, np.unique(normalized_all_sampled_depths)
    #assert False, [normalized_all_sampled_descriptors.shape, normalized_all_descriptors.shape]

    sum_of_squared_dists = []
    n_cluster_range = list(range(1, max_cluster))
    for n_clusters in n_cluster_range:
        algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
        algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
        squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
        objective = squared_distances.sum()
        sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
        if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
            break
    #assert False, labels.shape
    num_labels = np.max(n_clusters) + 1
    num_descriptors_per_image = [num_patches[0]*num_patches[1] for num_patches in num_patches_list]
    labels_per_image = np.split(labels, np.cumsum(num_descriptors_per_image))
    #assert False, normalized_all_sampled_descriptors.shape
    if use_gt_sal:
        votes = np.zeros(num_labels)
        for image_labels, saliency_map in zip(labels_per_image, saliency_maps_list):
            for label in range(num_labels):
                label_saliency = saliency_map[image_labels[:, 0] == label].mean()
                if label_saliency > thresh:
                    votes[label] += 1
        salient_labels = np.where(votes >= np.ceil(num_images * votes_percentage / 100))
        
    if save_png:
        cmap = 'jet' if num_labels > 10 else 'tab10'
        for i, (num_patches, label_per_image) in enumerate(zip(num_patches_list, labels_per_image)):
            mask = np.isin(label_per_image, salient_labels)
            label_per_image[~mask] = -1
            image_path = feat_files[i][:-3]
            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(label_per_image.reshape(num_patches), vmin=0, vmax=num_labels-1, cmap=cmap)
            #assert False, num_patches
            fig.savefig(image_path+f'_clustering_{max_cluster}_{depth_ratio}_{pixel_ratio}_{pts_ratio}_{use_gt_dino}_{use_gt_sal}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            original_path = image_path.replace('dinos', 'images')
            original_image = cv2.imread(original_path+'.jpg')
            cluster_mask = cv2.imread(image_path+f'_clustering_{max_cluster}_{depth_ratio}_{pixel_ratio}_{pts_ratio}_{use_gt_dino}_{use_gt_sal}.png')
            #assert False, [cluster_mask.shape
            cluster_mask = cv2.resize(cluster_mask, (original_image.shape[1], original_image.shape[0]), 0, 0, interpolation=cv2.INTER_NEAREST)
            #assert False, [original_image.shape, cluster_mask.shape, np.max(original_image), np.max(cluster_mask)]
            dst = cv2.addWeighted(original_image, 0.5, cluster_mask, 0.5, 0)
            cv2.imwrite(image_path+f'_blended_{max_cluster}_{depth_ratio}_{pixel_ratio}_{pts_ratio}_{use_gt_dino}_{use_gt_sal}.png', dst)
            if use_gt_sal:
                cv2.imwrite(image_path+f'_sal.png', saliency_maps_list[i].reshape((dst.shape[0], dst.shape[1]))*255.)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster sems')
    parser.add_argument('--root_dir', type=str, required=True, help='The root dir of image sets.')
    parser.add_argument('--max_cluster', type=int, required=True, help='how many clusters')
    parser.add_argument('--depth_ratio', type=float, default=0, help="how much depth information to use")
    parser.add_argument('--pixel_ratio', type=float, default=0, help="how much pixel information to use")
    parser.add_argument('--pts_ratio', type=float, default=0, help="how much 3D points information to use")
    parser.add_argument('--use_gt_dino', action="store_true", help="whether use gt dino feature without reconstruction")
    parser.add_argument('--use_gt_sal', action="store_true", help="whether use gt saliency feature without reconstruction")
    parser.add_argument('--votes_percentage', default=75, type=int, help="percentage of votes needed for a cluster to "
    "be considered salient.")
    parser.add_argument('--thresh', default=0.065, type=float, help='saliency maps threshold to distinguish fg / bg.') 
    

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

    args = parser.parse_args()

    feats = load_feats(args.root_dir, sample_interval=100, max_cluster=args.max_cluster, elbow=0.975, use_gt_dino=args.use_gt_dino, use_gt_sal=args.use_gt_sal, depth_ratio=args.depth_ratio, pixel_ratio=args.pixel_ratio,
        pts_ratio=args.pts_ratio,
        load_size=args.load_size, stride=args.stride, model_type=args.model_type, facet=args.facet, layer=args.layer, bin=args.bin, remove_outliers=args.remove_outliers,
        votes_percentage=args.votes_percentage, thresh=args.thresh)