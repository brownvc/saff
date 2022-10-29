import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
from kornia import create_meshgrid

from render_utils import *
from run_nerf_helpers import *
from load_llff import *

sys.path.append("../../dino_utils")
from extractor import *
from cosegmentation import *
from sklearn.decomposition import PCA, IncrementalPCA
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',

                        help='input data directory')
    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')
    parser.add_argument("--render_slowmo_full", action="store_true",
                        help="render space-time interpolation and store all information")
    parser.add_argument("--render_pcd_color", action="store_true",
                        help="render colored point cloud")
    parser.add_argument("--render_pcd_cluster", action="store_true",
                        help="render clustered point cloud")
    parser.add_argument("--render_pcd_cluster_3D", action="store_true",
                        help="render clustered point cloud in 3D")
    parser.add_argument("--render_sal_3D", action="store_true",
                        help="render saliency in 3D")
    parser.add_argument("--cluster_pcd", action="store_true",
                        help="cluster point cloud in 3D ")
    parser.add_argument("--cluster_2D", action="store_true",
                        help="cluster on 2D rendered result ")
    parser.add_argument("--render_mode", action="store_true",
                        help="generation decomposition result")
    parser.add_argument("--cluster_finch", action="store_true", help="cluster point cloud in 3D finch")
    parser.add_argument("--load_algo", type=str,
                        help="clustering algorithm to use")
    parser.add_argument("--n_cluster", type=int,
                        help="how many clusters to use")

    parser.add_argument("--render_2D", action="store_true",
                        help="Store 2D rendering result")


    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 512x288')
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*128, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_bt", action='store_true', 
                        help='render bullet time')

    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--target_idx", type=int, default=10, 
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')

    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--depth_full", action='store_true', 
                        help='enforce depth loss on full depth map instead of dynamic map only')

    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')

    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')

    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')

    # add dino args
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
    parser.add_argument('--load_dino_size', default=128, type=int, help='load size of the input image.')
    parser.add_argument('--dino_coe', default=0.0, type=float, help='weight of the feature loss.')
    parser.add_argument('--sal_coe', default=0.0, type=float, help='weight of the saliency loss.')
    
    parser.add_argument("--shallow_dino", action='store_true', 
                        help='use one layer as dino head')
    parser.add_argument("--prep_dino", action='store_true', 
                        help='preprocess dino as in D3F')              
    parser.add_argument("--use_tanh", action='store_true',
                        help="use tanh as in D3F")  
    parser.add_argument("--n_components", default=64, type=int, help="pca components")  
    
    
    parser.add_argument("--dino_weight", default=1, type=float, help="dino feature importance")  
    parser.add_argument("--flow_weight", default=0, type=float, help="flow feature importance")  

    
    #args.use_multi_dino
    parser.add_argument("--use_multi_dino", action="store_true", help="whether use multi-resolution dino")  

    parser.add_argument("--decay_extra", action="store_true", help="whether decay dino and sal L2 losses") 

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, depths, masks, poses, bds, \
        render_poses, ref_c2w, motion_coords = load_llff_data(args.datadir, 
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        
        if args.use_multi_dino:
            # a version of multiresolution
            assert args.dino_coe >0, "has to make sure dino is being used"
            assert args.prep_dino, "Has to make sure dim is small enough other wise explode cpu/gpu"
            #assert args.sal_coe >0, "has to make sure saliency is smooth as well"
            extractor = ViTExtractor(args.model_type, args.stride, device=device)
            saliency_extractor = extractor
            
            # first calculate all bbox for each image to compute
            # then as in the old time, group into batch
            start = 0
            N_samples = [0, 0.5, 1]
            coords = []
            height, width = images.shape[1], images.shape[2]
            if height < width:
                dwidth = int(args.load_dino_size / float(height) * width)
                dheight = args.load_size
            else:
                dheight = int(args.load_dino_size / float(width) * height)
                dwidth = args.load_size
            
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
            #assert False, coords
            start = None
            
            
            feats = None
            #torch.zeros((len(images), images[0].shape[0], images[0].shape[1], args.n_components))
            counter = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], 1))     
            sals = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], 1))            
            for [image_id, start_height, start_width, end_height, end_width] in tqdm(coords):
                batch = images[image_id:image_id+1, start_height:end_height, start_width:end_width]
                batch = torch.tensor(batch).permute(0, 3, 1, 2)
                batch = F.interpolate(batch, size=(dheight, dwidth), mode='nearest')
                with torch.no_grad():
                    feat_raw = extractor.extract_descriptors(batch.to(device), args.layer, args.facet, args.bin)
                    feat_raw = feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                    feat_raw = F.interpolate(feat_raw, size=(end_height - start_height, end_width - start_width), mode='nearest')
                    if feats is None:
                        feats = torch.zeros((len(images), images[0].shape[0], images[0].shape[1], feat_raw.shape[1]))
                    #assert False, [feats.shape, feat_raw.shape]
                    feats[image_id:image_id+1, start_height:end_height, start_width:end_width] += feat_raw.permute(0, 2, 3, 1)
                    counter[image_id:image_id+1, start_height:end_height, start_width:end_width] += 1
                    sal_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                    sal_raw = sal_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                    sal_raw = F.interpolate(sal_raw, size=(end_height - start_height, end_width - start_width), mode='nearest')
                    sals[image_id:image_id+1, start_height:end_height, start_width:end_width] += sal_raw.permute(0, 2, 3, 1)
                   
            feats /= 1e-16 + counter
            sals /= 1e-16 + counter
            
            feats = F.normalize(feats, p=2, eps=1e-12, dim=-1).cpu()
            counter = None
            sals = sals.cpu()

            #print("I am done")
            pca = PCA(n_components=args.n_components).fit(feats.view(-1, feats.shape[-1])[::100])
            #print("I am done")
            #print("I am done")
            split_idxs = np.array([images.shape[1] * images.shape[2] for _ in range(images.shape[0])])
            split_idxs = np.cumsum(split_idxs)
            feats = np.split(feats.view(-1, feats.shape[-1]).numpy(), split_idxs[:-1], axis=0)
            #print("I am done")
            feats = [pca.transform(feat) for feat in feats]
            feats = torch.from_numpy(np.concatenate(feats, axis=0)).view(images.shape[0], images.shape[1], images.shape[2], -1)
            #assert False, [len(num_patches_list), pca_feats.shape]
            
            '''
            #print("I am done")
            # visualize in PCA           
            pca = PCA(n_components=3).fit(feats[0].view(-1, feats.shape[-1]).cpu().numpy())
            #print("I am done")
            pca_feats = pca.transform(feats[0].view(-1, feats.shape[-1]).cpu().numpy())
            #print("I am done")
            pca_feats = pca_feats.reshape((images[0].shape[0], images[0].shape[1], pca_feats.shape[-1]))
            for comp_idx in range(3):
                comp = pca_feats[:, :, comp_idx]
                comp_min = comp.min(axis=(0, 1))
                comp_max = comp.max(axis=(0, 1))
                comp_img = (comp - comp_min) / (comp_max - comp_min)
                pca_feats[..., comp_idx] = comp_img
            cv2.imwrite("test_gt.png", pca_feats * 255.)
            cv2.imwrite("test_sal.png", sals.numpy()[0] * 255.)
            assert False
            '''
            print("Loaded dino features ", feats.shape)
            start = None
            pca = None

        elif args.dino_coe > 0:
            extractor = ViTExtractor(args.model_type, args.stride, device=device)
            if args.sal_coe > 0:
                saliency_extractor = extractor
            
            
            # have to do this in batch otherwise blows the gpu
            start = 0
            feats = None
            sals = None
            num_patches_list = []
            while start < images.shape[0]:
                #print(start)
                batch = images[start:min(images.shape[0], start + args.dino_batch)]
                batch = torch.tensor(batch).permute(0, 3, 1, 2)
                height, width = batch.shape[2], batch.shape[3]
                if height < width:
                    dwidth = int(args.load_dino_size / float(height) * width)
                    dheight = args.load_size
                else:
                    dheight = int(args.load_dino_size / float(width) * height)
                    dwidth = args.load_size
                batch = F.interpolate(batch, size=(dheight, dwidth), mode='nearest')
                #print(batch.shape)
                with torch.no_grad():
                    
                    feat_raw = extractor.extract_descriptors(batch.to(device), args.layer, args.facet, args.bin)
                    feat_raw = feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                    if args.sal_coe > 0:
                        #assert False, batch.shape
                        sal_raw = saliency_extractor.extract_saliency_maps(batch.to(device))
                        sal_raw = sal_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2)
                        #assert False, [sal_raw.shape, feat_raw.shape]
                    num_patches_list += [extractor.num_patches]*feat_raw.shape[0]
                #feat_raw = F.interpolate(feat_raw.view(batch.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).permute(0, 3, 1, 2), (height, width), mode="bilinear")
                if feats is None:
                    feats = feat_raw.permute(0, 2, 3, 1).detach().cpu()
                    if args.sal_coe > 0:
                        sals = sal_raw.permute(0, 2, 3, 1).detach().cpu()
                else:         
                    feats = torch.cat((feats, feat_raw.permute(0, 2, 3, 1).detach().cpu()), dim=0)
                    if args.sal_coe > 0:
                        sals = torch.cat((sals, sal_raw.permute(0, 2, 3, 1).detach().cpu()), dim = 0)
                batch = batch.cpu()
                feat_raw = feat_raw.detach().cpu()
                if args.sal_coe > 0:
                    sal_raw = sal_raw.detach().cpu()
                #assert False, [feats.shape, sals.shape]
                start += args.dino_batch
            if args.prep_dino:
                #feat_norm = torch.linalg.norm(feats, dim=-1, keepdim=True)
                #assert False, [torch.max(feats), torch.min(feats), feat_norm.shape]
                #feats = feats / feat_norm
                feats = torch.nn.functional.normalize(feats, p=2.0, dim=-1, eps=1e-12, out=None)
                #assert False, [torch.max(feats), torch.min(feats),]
                #assert False, feats.shape
                old_shape = feats.shape
                feats = feats.view(-1, feats.shape[-1])
                pca = PCA(n_components=args.n_components).fit(feats)
                pca_feats = pca.transform(feats)
                #assert False, [len(num_patches_list), pca_feats.shape]
                split_idxs = np.array([num_patches[0] * num_patches[1] for num_patches in num_patches_list])
                split_idxs = np.cumsum(split_idxs)
                pca_per_image = np.split(pca_feats, split_idxs[:-1], axis=0)
                feats = torch.from_numpy(np.stack(pca_per_image, axis=0)).view(old_shape[0], old_shape[1], old_shape[2], -1)
                '''
                # visualize in PCA
                feats = torch.nn.functional.interpolate(feats.permute(0, 3, 1, 2), (images[0].shape[0], images[0].shape[1]), mode=
                'nearest').permute(0, 2,3 ,1)
                assert False, feats.shape
                pca = PCA(n_components=3).fit(feats[0].view(-1, feats.shape[-1]).cpu().numpy())
                pca_feats = pca.transform(feats[0].view(-1, feats.shape[-1]).cpu().numpy())
                pca_feats = pca_feats.reshape((images[0].shape[0], images[0].shape[1], pca_feats.shape[-1]))
                for comp_idx in range(3):
                    comp = pca_feats[:, :, comp_idx]
                    comp_min = comp.min(axis=(0, 1))
                    comp_max = comp.max(axis=(0, 1))
                    comp_img = (comp - comp_min) / (comp_max - comp_min)
                    pca_feats[..., comp_idx] = comp_img
                cv2.imwrite("test_gt.png", pca_feats * 255.)
                assert False, "Pause"
                #assert False, "Debugging! not finished at all; model architecture unchanged for example; better to preprocess before training"
                '''
            print("Loaded dino features ", feats.shape)
            start = None
            
        
        i_test = []
        i_val = [] #i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    # args.expname = args.expname + '_sigma_rgb-%.2f'%(args.sigma_rgb) \
                # + '_use-rgb-w_' + str(args.use_rgb_w) + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(feats.shape[-1] if args.dino_coe > 0 else 0, args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)


    if args.render_bt:
        assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER VIEW INTERPOLATION')      
        render_poses = torch.Tensor(render_poses).to(device)
        print('target_idx ', target_idx)

        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0

        testsavedir = os.path.join(basedir, expname, 
                                'render-spiral-frame-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_bullet_time(render_poses, img_idx_embed, num_img, hwf, 
                               args.chunk, render_kwargs_test, 
                               gt_imgs=images, savedir=testsavedir, 
                               render_factor=args.render_factor)

        return

    if args.render_lockcam_slowmo:
        assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER TIME INTERPOLATION')
        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        print('target_idx ', target_idx)

        testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_lockcam_slowmo(ref_c2w, num_img, hwf, 
                            args.chunk, render_kwargs_test, 
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor,
                            target_idx=target_idx)

            return 

    if args.render_slowmo_bt:
        assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER SLOW MOTION') 
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10

        with torch.no_grad():

            testsavedir = os.path.join(basedir, expname, 
                                    'render-slowmo_bt_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            images = torch.Tensor(images)#.to(device)

            print('render poses shape', render_poses.shape)
            render_slowmo_bt(depths, render_poses, bt_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
            # print('Done rendering', i,testsavedir)

        return
    if args.render_slowmo_full:
        assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER SLOW MOTION') 
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10

        with torch.no_grad():

            testsavedir = os.path.join(basedir, expname, 
                                    'render-slowmo_full_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            images = torch.Tensor(images)#.to(device)

            print('render poses shape', render_poses.shape)
            render_slowmo_full(depths, render_poses, bt_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
            # print('Done rendering', i,testsavedir)

        return
    if args.render_pcd_color:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER pcd color')
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'render-pcd_color-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            #assert False, "parameters not decided!"
            render_pcd_color(render_poses, bt_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
        return
    
    if args.render_pcd_cluster:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER pcd cluster')
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'render-pcd_cluster-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        #assert args.load_algo != '' and os.path.exists(args.load_algo), "must have valid cluster stored"
        #n_clusters = int(args.load_algo.split("/")[-1].split("_")[1])
        #assert False, [load_algo, n_clusters]
        #assert False, feats[0].shape
        #algorithm = faiss.Kmeans(d=(3+feats[0].shape[-1]), k=n_clusters, niter=300, nredo=10)
        #centroids = np.load(args.load_algo)
        #sample_data = np.load(args.load_algo.replace('centroids', 'sample'))
        #algorithm.centroids = centroids
        #algorithm.train(sample_data.astype(np.float32), init_centroids=centroids) 
        #assert np.sum(algorithm.centroids - centroids) == 0, "centroids are not the same"
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index("./logs/experiment_Jumping_sal_F00-30/cluster_pcd-010_path_360001/large.index"))
        salient_labels = np.load("./logs/experiment_Jumping_sal_F00-30/cluster_pcd-010_path_360001/saliency.npy")
        
        render_kwargs_test["N_samples"] = 64
        with torch.no_grad():
            #assert False, "parameters not decided!"
            render_pcd_cluster(index, salient_labels, render_poses, bt_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
        return
    if args.cluster_pcd:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('cluster in 3D')
        assert args.use_tanh, "Need to make sure dino feature falls in between -1 and 1"
        assert (args.dino_coe > 0) and (args.sal_coe > 0), "must have both dino head and saliency head"
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        #assert False, render_poses.shape
        #bt_poses = create_bt_poses(hwf) 
        #bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'cluster_pcd-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        #assert args.load_algo != '' and os.path.exists(args.load_algo), "must have valid cluster stored"
        #assert False, [load_algo, n_clusters]
        #assert False, feats[0].shape
        
        #algorithm = faiss.Kmeans(d=(3+feats[0].shape[-1]), k=args.n_cluster, niter=300, nredo=10)
        #centroids = np.load(args.load_algo)
        #sample_data = np.load(args.load_algo.replace('centroids', 'sample'))
        #algorithm.centroids = centroids
        #algorithm.train(sample_data.astype(np.float32), init_centroids=centroids) 
        #assert np.sum(algorithm.centroids - centroids) == 0, "centroids are not the same"
        #salient_labels = np.load(args.load_algo.replace('centroids', 'salient'))
        
            
        with torch.no_grad():
            #assert False, "parameters not decided!"
            res = faiss.StandardGpuResources()
            try:
                index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "large.index")))
                salient_labels = np.load(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "saliency.npy"))
            except:
                index = None
                salient_labels = None
            cluster_pcd(render_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            dino_weight=args.dino_weight,
                            flow_weight=args.flow_weight,
                            index = index,
                            salient_labels = salient_labels,
                            savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            )
        return
    if args.cluster_2D:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('cluster in 2D')
        assert args.use_tanh, "Need to make sure dino feature falls in between -1 and 1"
        assert (args.dino_coe > 0) and (args.sal_coe > 0), "must have both dino head and saliency head"
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        #assert False, render_poses.shape
        #bt_poses = create_bt_poses(hwf) 
        #bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'cluster_2D-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        #assert args.load_algo != '' and os.path.exists(args.load_algo), "must have valid cluster stored"
        #assert False, [load_algo, n_clusters]
        #assert False, feats[0].shape
        
        #algorithm = faiss.Kmeans(d=(3+feats[0].shape[-1]), k=args.n_cluster, niter=300, nredo=10)
        #centroids = np.load(args.load_algo)
        #sample_data = np.load(args.load_algo.replace('centroids', 'sample'))
        #algorithm.centroids = centroids
        #algorithm.train(sample_data.astype(np.float32), init_centroids=centroids) 
        #assert np.sum(algorithm.centroids - centroids) == 0, "centroids are not the same"
        #salient_labels = np.load(args.load_algo.replace('centroids', 'salient'))
        
            
        with torch.no_grad():
            #assert False, "parameters not decided!"
            res = faiss.StandardGpuResources()
            try:
                index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(os.path.join(basedir, expname, 'cluster_2D-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "large.index")))
                
                salient_labels = np.load(os.path.join(basedir, expname, 'cluster_2D-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "salient.npy"))
                label_mapper = pickle.load(open(os.path.join(basedir, expname, 'cluster_2D-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "label_mapper.pkl"), "rb"))
            
            except:
                index = None
                salient_labels = None
                label_mapper = None
            if args.render_mode:
                assert index is not None and salient_labels is not None and label_mapper is not None
            cluster_2D(render_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            dino_weight=args.dino_weight,
                            flow_weight=args.flow_weight,
                            index = index,
                            salient_labels = salient_labels,
                            label_mapper = label_mapper,
                            render_mode = args.render_mode,
                            savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            )
        return
    if args.render_2D:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('render in 2D')
        #assert args.use_tanh, "Need to make sure dino feature falls in between -1 and 1"
        #assert (args.dino_coe > 0) and (args.sal_coe > 0), "must have both dino head and saliency head"
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        #assert False, render_poses.shape
        #bt_poses = create_bt_poses(hwf) 
        #bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'render_2D-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        #assert args.load_algo != '' and os.path.exists(args.load_algo), "must have valid cluster stored"
        #assert False, [load_algo, n_clusters]
        #assert False, feats[0].shape
        
        #algorithm = faiss.Kmeans(d=(3+feats[0].shape[-1]), k=args.n_cluster, niter=300, nredo=10)
        #centroids = np.load(args.load_algo)
        #sample_data = np.load(args.load_algo.replace('centroids', 'sample'))
        #algorithm.centroids = centroids
        #algorithm.train(sample_data.astype(np.float32), init_centroids=centroids) 
        #assert np.sum(algorithm.centroids - centroids) == 0, "centroids are not the same"
        #salient_labels = np.load(args.load_algo.replace('centroids', 'salient'))
        
            
        with torch.no_grad():
            #assert False, "parameters not decided!"
            #res = faiss.StandardGpuResources()
            #try:
            #    index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(os.path.join(basedir, expname, 'cluster_2D-%03d'%\
            #                    target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "large.index")))
            #    
            #    salient_labels = np.load(os.path.join(basedir, expname, 'cluster_2D-%03d'%\
            #                    target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "salient.npy"))
            #    label_mapper = pickle.load(open(os.path.join(basedir, expname, 'cluster_2D-%03d'%\
            #                    target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "label_mapper.pkl"), "rb"))
            
            #except:
            #    index = None
            #    salient_labels = None
            #    label_mapper = None
            #if args.render_mode:
            #    assert index is not None and salient_labels is not None and label_mapper is not None
            render_2D(render_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            return_sem=args.dino_coe != 0
                            )
        return
    if args.render_pcd_cluster_3D:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER pcd cluster in 3D')
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'render-pcd_cluster_3D-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        #assert args.load_algo != '' and os.path.exists(args.load_algo), "must have valid cluster stored"
        #n_clusters = int(args.load_algo.split("/")[-1].split("_")[1])
        #assert False, [load_algo, n_clusters]
        #assert False, feats[0].shape
        #algorithm = faiss.Kmeans(d=(3+feats[0].shape[-1]), k=n_clusters, niter=300, nredo=10)
        #centroids = np.load(args.load_algo)
        #sample_data = np.load(args.load_algo.replace('centroids', 'sample'))
        #algorithm.centroids = centroids
        #algorithm.train(sample_data.astype(np.float32), init_centroids=centroids) 
        #assert np.sum(algorithm.centroids - centroids) == 0, "centroids are not the same"
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "large.index")))
        #           "./logs/experiment_Jumping_sal_F00-30/cluster_pcd-010_path_360001/large.index"))
        salient_labels = np.load(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "saliency.npy"))
        small_index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "small.index")))
        #dino_normalizer = torch.load(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "dino_normalizer.pt"))
        #small_index = pickle.load(open(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "save.pkl"), "rb"))
        #point_normalizer = torch.load(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "point_normalizer.pt"))
        #render_kwargs_test["N_samples"] = 64
        #assert False, render_poses.shape
        num_img = len(render_poses)
        assert num_img == 24
        for _ in range(num_img-1):
            render_poses = np.concatenate([render_poses, render_poses[:1,...]], axis=0)
        for i in range(11):
            render_poses = np.concatenate([render_poses, render_poses[i+1:i+2, ...]], axis=0)
        #render_poses = torch.cat([render_poses] + [render_poses[:1, :, :]]*num_img + render_poses[:12, :, :], dim=0)
        img_idx_embeds = list(range(num_img)) + list(range(1, num_img)) + [0]*11
        #assert False, [render_poses.shape, img_idx_embeds]
        img_idx_embeds = [t/float(num_img) * 2. - 1.0 for t in img_idx_embeds]
        with torch.no_grad():
            #assert False, "parameters not decided!"
            render_pcd_cluster_3D(index, small_index, salient_labels, render_poses, img_idx_embeds, 
                            hwf, args.chunk, render_kwargs_test, dino_weight=args.dino_weight,
                            flow_weight=args.flow_weight,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
        return
    if args.cluster_finch:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('cluster in finch 3D')
        assert args.use_tanh, "Need to make sure dino feature falls in between -1 and 1"
        assert (args.dino_coe > 0) and (args.sal_coe > 0), "must have both dino head and saliency head"
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        #assert False, render_poses.shape
        #bt_poses = create_bt_poses(hwf) 
        #bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'cluster_finch-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        #assert args.load_algo != '' and os.path.exists(args.load_algo), "must have valid cluster stored"
        #assert False, [load_algo, n_clusters]
        #assert False, feats[0].shape
        
        #algorithm = faiss.Kmeans(d=(3+feats[0].shape[-1]), k=args.n_cluster, niter=300, nredo=10)
        #centroids = np.load(args.load_algo)
        #sample_data = np.load(args.load_algo.replace('centroids', 'sample'))
        #algorithm.centroids = centroids
        #algorithm.train(sample_data.astype(np.float32), init_centroids=centroids) 
        #assert np.sum(algorithm.centroids - centroids) == 0, "centroids are not the same"
        #salient_labels = np.load(args.load_algo.replace('centroids', 'salient'))
        
            
        with torch.no_grad():
            
            cluster_finch(render_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            dino_weight=args.dino_weight,
                            flow_weight=args.flow_weight, 
                            savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            )
        return
    if args.render_sal_3D:
        #assert False, "axis may be wrong due to saliency channel!!!"
        print('RENDER saliency in 3D')
        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10
        
        testsavedir = os.path.join(basedir, expname, 
                                'render-sal_3D-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #assert False, testsavedir
        os.makedirs(testsavedir, exist_ok=True)
        #assert args.load_algo != '' and os.path.exists(args.load_algo), "must have valid cluster stored"
        #n_clusters = int(args.load_algo.split("/")[-1].split("_")[1])
        #assert False, [load_algo, n_clusters]
        #assert False, feats[0].shape
        #algorithm = faiss.Kmeans(d=(3+feats[0].shape[-1]), k=n_clusters, niter=300, nredo=10)
        #centroids = np.load(args.load_algo)
        #sample_data = np.load(args.load_algo.replace('centroids', 'sample'))
        #algorithm.centroids = centroids
        #algorithm.train(sample_data.astype(np.float32), init_centroids=centroids) 
        #assert np.sum(algorithm.centroids - centroids) == 0, "centroids are not the same"
        #res = faiss.StandardGpuResources()
        #index = faiss.index_cpu_to_gpu(res, 0, faiss.read_index(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "large.index")))
        #           "./logs/experiment_Jumping_sal_F00-30/cluster_pcd-010_path_360001/large.index"))
        #salient_labels = np.load(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "saliency.npy"))
        
        #dino_normalizer = torch.load(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "dino_normalizer.pt"))
        #clustering = pickle.load(open(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "save.pkl"), "rb"))
        #point_normalizer = torch.load(os.path.join(basedir, expname, 'cluster_pcd-%03d'%\
        #                        target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start), "point_normalizer.pt"))
        #render_kwargs_test["N_samples"] = 64
        #assert False, render_poses.shape
        num_img = len(render_poses)
        assert num_img == 24
        for _ in range(num_img-1):
            render_poses = np.concatenate([render_poses, render_poses[:1,...]], axis=0)
        for i in range(11):
            render_poses = np.concatenate([render_poses, render_poses[i+1:i+2, ...]], axis=0)
        #render_poses = torch.cat([render_poses] + [render_poses[:1, :, :]]*num_img + render_poses[:12, :, :], dim=0)
        img_idx_embeds = list(range(num_img)) + list(range(1, num_img)) + [0]*11
        #assert False, [render_poses.shape, img_idx_embeds]
        img_idx_embeds = [t/float(num_img) * 2. - 1.0 for t in img_idx_embeds]
        with torch.no_grad():
            #assert False, "parameters not decided!"
            render_sal_3D(render_poses, img_idx_embeds, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # Move training data to GPU
    images = torch.Tensor(images)#.to(device)
    depths = torch.Tensor(depths)#.to(device)
    masks = 1.0 - torch.Tensor(masks).to(device)

    poses = torch.Tensor(poses).to(device)

    #N_iters = 2000 * 1000 #1000000
    N_iters = 370 * 1000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda() # (H, W, 2)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])
    
    decay_iteration = max(args.decay_iteration, 
                          args.end_frame - args.start_frame)
    decay_iteration = min(decay_iteration, 250)

    chain_bwd = 0

    for i in range(start, N_iters):
        chain_bwd = 1 - chain_bwd
        time0 = time.time()
        print('expname ', expname, ' chain_bwd ', chain_bwd, 
             ' lindisp ', args.lindisp, ' decay_iteration ', decay_iteration)
        print('Random FROM SINGLE IMAGE')
        # Random from one image
        img_i = np.random.choice(i_train)

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        target = images[img_i].cuda()
        pose = poses[img_i, :3,:4]
        depth_gt = depths[img_i].cuda()
        hard_coords = torch.Tensor(motion_coords[img_i]).cuda()
        mask_gt = masks[img_i].cuda()
        if args.dino_coe > 0:
            feat_gt = feats[img_i].cuda()
            if args.sal_coe > 0:
                sal_gt = sals[img_i].cuda()

        if img_i == 0:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, 
                                                args.start_frame, fwd=True)
            flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
        elif img_i == num_img - 1:
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, 
                                                args.start_frame, fwd=False)
            flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
        else:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, 
                                                img_i, args.start_frame, 
                                                fwd=True)
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, 
                                                img_i, args.start_frame, 
                                                fwd=False)

        # # ======================== TEST 
        TEST = False
        if TEST:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            print('CHECK DEPTH and FLOW and exiting')
            print(images[img_i].shape)
            print(flow_fwd.shape, img_i)

            warped_im2 = warp_flow(images[img_i + 1].cpu().numpy(), flow_fwd)
            warped_im0 = warp_flow(images[img_i - 1].cpu().numpy(), flow_bwd)
            mask_gt = masks[img_i].cpu().numpy()

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 3, 1)
            plt.imshow(target.cpu().numpy())
            plt.subplot(2, 3, 4)
            plt.imshow(depth_gt.cpu().numpy(), cmap='jet') 

            plt.subplot(2, 3, 2)
            plt.imshow(flow_to_image(flow_fwd)/255. * fwd_mask[..., np.newaxis])

            plt.subplot(2, 3, 3)
            plt.imshow(flow_to_image(flow_bwd)/255. * bwd_mask[..., np.newaxis])

            plt.subplot(2, 3, 5)
            plt.imshow(mask_gt, cmap='gray')

            cv2.imwrite('im_%d.jpg'%(img_i),
                        np.uint8(np.clip(target.cpu().numpy()[:, :, ::-1], 0, 1) * 255))
            cv2.imwrite('im_%d_warp.jpg'%(img_i + 1), 
                        np.uint8(np.clip(warped_im2[:, :, ::-1], 0, 1) * 255))
            cv2.imwrite('im_%d_warp.jpg'%(img_i - 1), 
                        np.uint8(np.clip(warped_im0[:, :, ::-1], 0, 1) * 255))
            plt.savefig('depth_flow_%d.jpg'%img_i)
            sys.exit()

        #  END OF TEST
        flow_fwd = torch.Tensor(flow_fwd).cuda()
        fwd_mask = torch.Tensor(fwd_mask).cuda()
    
        flow_bwd = torch.Tensor(flow_bwd).cuda()
        bwd_mask = torch.Tensor(bwd_mask).cuda()
        # more correct way for flow loss
        flow_fwd = flow_fwd + uv_grid
        flow_bwd = flow_bwd + uv_grid

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

            if args.use_motion_mask and i < decay_iteration * 1000:
                print('HARD MINING STAGE !')
                num_extra_sample = args.num_extra_sample
                print('num_extra_sample ', num_extra_sample)
                select_inds_hard = np.random.choice(hard_coords.shape[0], 
                                                    size=[min(hard_coords.shape[0], 
                                                        num_extra_sample)], 
                                                    replace=False)  # (N_rand,)
                select_inds_all = np.random.choice(coords.shape[0], 
                                                size=[N_rand], 
                                                replace=False)  # (N_rand,)

                select_coords_hard = hard_coords[select_inds_hard].long()
                select_coords_all = coords[select_inds_all].long()

                select_coords = torch.cat([select_coords_all, select_coords_hard], 0)

            else:
                select_inds = np.random.choice(coords.shape[0], 
                                            size=[N_rand], 
                                            replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
            
            rays_o = rays_o[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_rgb = target[select_coords[:, 0], 
                                select_coords[:, 1]]  # (N_rand, 3)
            if args.dino_coe > 0:
                
                target_feat = feat_gt[(select_coords[:, 0] * (feats.shape[1] / float(images.shape[1]) )).long(),
                                (select_coords[:, 1] * (feats.shape[2]/ float(images.shape[2]) )).long()]
                if args.sal_coe > 0:
                    target_sal = sal_gt[(select_coords[:, 0] * (sals.shape[1]/ float(images.shape[1]) )).long(),
                                (select_coords[:, 1] *( sals.shape[2]/ float(images.shape[2]) )).long()]
                    
            target_depth = depth_gt[select_coords[:, 0], 
                                select_coords[:, 1]]
            target_mask = mask_gt[select_coords[:, 0], 
                                select_coords[:, 1]].unsqueeze(-1)

            target_of_fwd = flow_fwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_fwd_mask = fwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

            target_of_bwd = flow_bwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_bwd_mask = bwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

        img_idx_embed = img_i/num_img * 2. - 1.0

        #####  Core optimization loop  #####
        if args.chain_sf and i > decay_iteration * 1000 * 2:
            chain_5frames = True
        else:
            chain_5frames = False

        print('chain_5frames ', chain_5frames, ' chain_bwd ', chain_bwd)

        ret = render(img_idx_embed, 
                     chain_bwd, 
                     chain_5frames,
                     num_img, H, W, focal, 
                     chunk=args.chunk, 
                     rays=batch_rays,
                     verbose=i < 10, retraw=True,
                     **render_kwargs_train)

        pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
        pose_prev = poses[max(img_i - 1, 0), :3,:4]

        render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, 
                                                            pose, pose_prev, 
                                                            H, W, focal, 
                                                            ret)

        optimizer.zero_grad()

        weight_map_post = ret['prob_map_post']
        weight_map_prev = ret['prob_map_prev']

        weight_post = 1. - ret['raw_prob_ref2post']
        weight_prev = 1. - ret['raw_prob_ref2prev']
        prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                + torch.mean(torch.abs(ret['raw_prob_ref2post'])))

        # dynamic rendering loss
        if i <= decay_iteration * 1000:
            # dynamic rendering loss
            render_loss = img2mse(ret['rgb_map_ref_dy'], target_rgb)
            render_loss += compute_mse(ret['rgb_map_post_dy'], 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1))
            render_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1))
            if args.dino_coe > 0:
                render_loss_dino = img2mse(ret['dino_map_ref_dy'], target_feat)
                render_loss_dino += compute_mse(ret['dino_map_post_dy'],
                                                target_feat,
                                                weight_map_post.unsqueeze(-1))
                render_loss_dino += compute_mse(ret['dino_map_prev_dy'],
                                                target_feat,
                                                weight_map_prev.unsqueeze(-1))
                if args.sal_coe > 0:
                    render_loss_sal = img2mse(ret["sal_map_ref_dy"], target_sal)
                    render_loss_sal += compute_mse(ret["sal_map_post_dy"],
                                                target_sal,
                                                weight_map_post.unsqueeze(-1))
                    render_loss_sal += compute_mse(ret["sal_map_prev_dy"],
                                                target_sal,
                                                weight_map_prev.unsqueeze(-1))
        else:
            print('only compute dynamic render loss in masked region')
            weights_map_dd = ret['weights_map_dd'].unsqueeze(-1).detach()

            # dynamic rendering loss
            render_loss = compute_mse(ret['rgb_map_ref_dy'], 
                                      target_rgb, 
                                      weights_map_dd)
            render_loss += compute_mse(ret['rgb_map_post_dy'], 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1) * weights_map_dd)
            render_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1) * weights_map_dd)
            if args.dino_coe > 0:
                
                render_loss_dino = compute_mse(ret["dino_map_ref_dy"],
                                                target_feat,
                                                weights_map_dd)
                render_loss_dino += compute_mse(ret["dino_map_post_dy"],
                                                target_feat,
                                                weight_map_post.unsqueeze(-1) * weights_map_dd)
                render_loss_dino += compute_mse(ret["dino_map_prev_dy"],
                                                target_feat,
                                                weight_map_prev.unsqueeze(-1) * weights_map_dd)    
                if args.sal_coe > 0:
                    render_loss_sal = compute_mse(ret["sal_map_ref_dy"],
                                                target_sal,
                                                weights_map_dd)
                    render_loss_sal += compute_mse(ret["sal_map_post_dy"],
                                                target_sal,
                                                weight_map_post.unsqueeze(-1) * weights_map_dd)
                    render_loss_sal += compute_mse(ret["sal_map_prev_dy"],
                                                target_sal,
                                                weight_map_prev.unsqueeze(-1) * weights_map_dd)

        # union rendering loss
        render_loss += img2mse(ret['rgb_map_ref'][:N_rand, ...], 
                               target_rgb[:N_rand, ...])
        if args.dino_coe > 0:
            render_loss_dino += img2mse(ret["dino_map_ref"][:N_rand, ...],
                                    target_feat[:N_rand, ...])
            if args.sal_coe > 0:
                render_loss_sal += img2mse(ret["sal_map_ref"][:N_rand, ...],
                                    target_sal[:N_rand, ...])

        sf_cycle_loss = args.w_cycle * compute_mae(ret['raw_sf_ref2post'], 
                                                   -ret['raw_sf_post2ref'], 
                                                   weight_post.unsqueeze(-1), dim=3) 
        sf_cycle_loss += args.w_cycle * compute_mae(ret['raw_sf_ref2prev'], 
                                                    -ret['raw_sf_prev2ref'], 
                                                    weight_prev.unsqueeze(-1), dim=3)
        
        # regularization loss
        render_sf_ref2prev = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
        render_sf_ref2post = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)

        sf_reg_loss = args.w_sf_reg * (torch.mean(torch.abs(render_sf_ref2prev)) \
                                    + torch.mean(torch.abs(render_sf_ref2post))) 

        divsor = i // (decay_iteration * 1000)

        decay_rate = 10

        if args.decay_depth_w:
            w_depth = args.w_depth/(decay_rate ** divsor)
        else:
            w_depth = args.w_depth

        if args.decay_optical_flow_w:
            w_of = args.w_optical_flow/(decay_rate ** divsor)
        else:
            w_of = args.w_optical_flow
        
        depth_loss = w_depth * compute_depth_loss(ret['depth_map_ref_dy'] if not args.depth_full else ret['depth_map_ref'], -target_depth)

        print('w_depth ', w_depth, 'w_of ', w_of)

        if img_i == 0:
            print('only fwd flow')
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
        elif img_i == num_img - 1:
            print('only bwd flow')
            flow_loss = w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)
        else:
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
            flow_loss += w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)

        # scene flow smoothness loss
        sf_sm_loss = args.w_sm * (compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    H, W, focal) \
                                + compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal))

        # scene flow least kinectic loss
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal)
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal)
        entropy_loss = args.w_entropy * torch.mean(-ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-8))

        # # ======================================  two-frames chain loss ===============================
        if chain_bwd:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_prev'], 
                                                          ret['raw_pts_ref'], 
                                                          ret['raw_pts_pp'], 
                                                          H, W, focal)

        else:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_post'], 
                                                          ret['raw_pts_pp'], 
                                                          ret['raw_pts_ref'], 
                                                          H, W, focal)

        if chain_5frames:
            print('5 FRAME RENDER LOSS ADDED') 
            render_loss += compute_mse(ret['rgb_map_pp_dy'], 
                                       target_rgb, 
                                       weights_map_dd)
            if args.dino_coe > 0:
                render_loss_dino += compute_mse(ret["dino_map_pp_dy"],
                                            target_feat,
                                            weights_map_dd)
                if args.sal_coe > 0:
                    render_loss_sal += compute_mse(ret["sal_map_pp_dy"],
                                            target_sal,
                                            weights_map_dd) 
        if args.dino_coe > 0:
            if args.decay_extra:
                w_dino = args.dino_coe/(decay_rate ** divsor)
            else:
                w_dino = args.dino_coe 
            render_loss_dino = w_dino * render_loss_dino
            if args.sal_coe > 0:
                if args.decay_extra:
                    w_sal = args.sal_coe/(decay_rate ** divsor)
                else:
                    w_sal = args.sal_coe
                render_loss_sal = w_sal * render_loss_sal
        loss = sf_reg_loss + sf_cycle_loss + \
               render_loss + (render_loss_dino if args.dino_coe > 0 else 0) + (render_loss_sal if args.sal_coe > 0 else 0) + flow_loss + \
               sf_sm_loss + prob_reg_loss + \
               depth_loss + entropy_loss 

        if args.dino_coe > 0:
            print('render_loss_dino ', render_loss_dino.item())
            if args.sal_coe > 0:
                print('render_loss_sal ', render_loss_sal.item())
        print('render_loss ', render_loss.item(), 
              ' bidirection_loss ', sf_cycle_loss.item(), 
              ' sf_reg_loss ', sf_reg_loss.item())
        print('depth_loss ', depth_loss.item(), 
              ' flow_loss ', flow_loss.item(), 
              ' sf_sm_loss ', sf_sm_loss.item())
        print('prob_reg_loss ', prob_reg_loss.item(),
              ' entropy_loss ', entropy_loss.item())
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

            print('Saved checkpoints at', path)


        if i % args.i_print == 0 and i > 0:
            writer.add_scalar("train/loss", loss.item(), i)
            
            writer.add_scalar("train/render_loss", render_loss.item(), i)
            if args.dino_coe > 0:
                writer.add_scalar("train/render_loss_dino", render_loss_dino.item(), i)
                if args.sal_coe > 0:
                    writer.add_scalar("train/render_loss_sal", render_loss_sal.item(), i)
            writer.add_scalar("train/depth_loss", depth_loss.item(), i)
            writer.add_scalar("train/flow_loss", flow_loss.item(), i)
            writer.add_scalar("train/prob_reg_loss", prob_reg_loss.item(), i)

            writer.add_scalar("train/sf_reg_loss", sf_reg_loss.item(), i)
            writer.add_scalar("train/sf_cycle_loss", sf_cycle_loss.item(), i)
            writer.add_scalar("train/sf_sm_loss", sf_sm_loss.item(), i)


        if i%args.i_img == 0:
            # img_i = np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            target_depth = depths[img_i] - torch.min(depths[img_i])

            # img_idx_embed = img_i/num_img * 2. - 1.0

            # if img_i == 0:
            #     flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, 
            #                                            args.start_frame, fwd=True)
            #     flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
            # elif img_i == num_img - 1:
            #     flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, 
            #                                            args.start_frame, fwd=False)
            #     flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
            # else:
            #     flow_fwd, fwd_mask = read_optical_flow(args.datadir, 
            #                                            img_i, args.start_frame, 
            #                                            fwd=True)
            #     flow_bwd, bwd_mask = read_optical_flow(args.datadir, 
            #                                            img_i, args.start_frame, 
            #                                            fwd=False)

            # flow_fwd_rgb = torch.Tensor(flow_to_image(flow_fwd)/255.)#.cuda()
            # writer.add_image("val/gt_flow_fwd", 
            #                 flow_fwd_rgb, global_step=i, dataformats='HWC')
            # flow_bwd_rgb = torch.Tensor(flow_to_image(flow_bwd)/255.)#.cuda()
            # writer.add_image("val/gt_flow_bwd", 
            #                 flow_bwd_rgb, global_step=i, dataformats='HWC')

            with torch.no_grad():
                ret = render(img_idx_embed, 
                             chain_bwd, False,
                             num_img, H, W, focal, 
                             chunk=1024*16, 
                             c2w=pose,
                             **render_kwargs_test)

                # pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
                # pose_prev = poses[max(img_i - 1, 0), :3,:4]
                # render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, pose, pose_prev, 
                #                                                     H, W, focal, ret, n_dim=2)

                # render_flow_fwd_rgb = torch.Tensor(flow_to_image(render_of_fwd.cpu().numpy())/255.)#.cuda()
                # render_flow_bwd_rgb = torch.Tensor(flow_to_image(render_of_bwd.cpu().numpy())/255.)#.cuda()
                
                writer.add_image("val/rgb_map_ref", torch.clamp(ret['rgb_map_ref'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_ref", normalize_depth(ret['depth_map_ref']), 
                                global_step=i, dataformats='HW')
                
                if args.dino_coe > 0 and args.sal_coe > 0:
                    writer.add_image("val/sal_map_ref", torch.clamp(ret["sal_map_ref"][..., 0], 0., 1.),
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_rig", torch.clamp(ret['rgb_map_rig'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_rig", normalize_depth(ret['depth_map_rig']), 
                                global_step=i, dataformats='HW')
                if args.dino_coe > 0 and args.sal_coe > 0:
                    writer.add_image("val/sal_map_rig", torch.clamp(ret['sal_map_rig'][..., 0], 0., 1.), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_ref_dy", torch.clamp(ret['rgb_map_ref_dy'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_ref_dy", normalize_depth(ret['depth_map_ref_dy']), 
                                global_step=i, dataformats='HW')
                if args.dino_coe > 0 and args.sal_coe > 0:
                    writer.add_image("val/sal_map_ref_dy", torch.clamp(ret['sal_map_ref_dy'][..., 0], 0., 1.), 
                                global_step=i, dataformats='HW')

                # writer.add_image("val/rgb_map_pp_dy", torch.clamp(ret['rgb_map_pp_dy'], 0., 1.), 
                                # global_step=i, dataformats='HWC')
                
                writer.add_image("val/gt_rgb", target, 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/monocular_disp", 
                                torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/weights_map_dd", 
                                 ret['weights_map_dd'], 
                                 global_step=i, 
                                 dataformats='HW')

            torch.cuda.empty_cache()

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()