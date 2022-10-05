from sklearn.metrics import adjusted_rand_score
from piq import ssim as compute_ssim
from piq import psnr as compute_psnr
import numpy as np
import cv2
import torch
import lpips

# gt: HxW numpy array
# pred: HxW numpy array
# mask: HxW numpy array
# if mask exists, means only care about foreground's cluster
def ARI(gt, pred, mask=None):
    if mask is None:
        return adjusted_rand_score(gt.flatten(), pred.flatten())
    return adjusted_rand_score(gt[mask], pred[mask])

LPIPS_loss = lpips.LPIPS().cuda()

# gt and pred: normalized to -1, 1
# double tensor on cuda
# gt: [BxNxHxW]
# pred: [BxNxHxW]
def loss_lpips(gt, pred):
    return LPIPS_loss(gt.float(), pred.float()).mean()

# gt and pred: normalized to -1, 1
# double tensor on cuda
# gt: [BxNxHxW]
# pred: [BxNxHxW]
def loss_psnr(gt, pred):
    return compute_psnr(gt/2.+0.5, pred/2.+0.5, data_range=1.)

# gt and pred: normalized to -1, 1
# double tensor on cuda
# gt: [BxNxHxW]
# pred: [BxNxHxW]
def loss_ssim(gt, pred):
    return compute_ssim(gt/2.+0.5, pred/2.+0.5, data_range=1.)


if __name__ == "__main__":
    
    
    # validate ARI 
    gt_mask = cv2.imread("../data/object_masks/Jumping/training_ 0.png", cv2.IMREAD_GRAYSCALE)
    fg_mask = np.zeros(gt_mask.shape).astype(bool)
    fg_mask[gt_mask != 0] = True
    print(gt_mask.shape, np.unique(gt_mask))    
    pred_mask = cv2.imread("../data/object_masks/Jumping/training_ 0.png", cv2.IMREAD_GRAYSCALE)
    print("ARI with itself: ", ARI(gt_mask, pred_mask))
    print("fg-ARI with itself: ", ARI(gt_mask, pred_mask, fg_mask))
    #pred_mask = cv2.imread("../data/object_masks/Jumping/nv_static_49.png", cv2.IMREAD_GRAYSCALE)
    #print("ARI with its spatial neighbor: ", ARI(gt_mask, pred_mask))
    pred_mask = cv2.imread("../data/object_masks/Jumping/nv_spatial_25.png", cv2.IMREAD_GRAYSCALE)
    print("ARI with its temporal neighbor: ", ARI(gt_mask, pred_mask))
    print("fg-ARI with its temporal neighbor: ", ARI(gt_mask, pred_mask, fg_mask))
    
    pred_mask = cv2.imread("../data/object_masks/Jumping/nv_static_57.png", cv2.IMREAD_GRAYSCALE)
    print("ARI with other frame the same scene, same time, different view: ", ARI(gt_mask, pred_mask))
    print("fg-ARI with other frame the same scene, same time, different view: ", ARI(gt_mask, pred_mask, fg_mask))
    pred_mask = cv2.imread("../data/object_masks/Jumping/nv_spatial_39.png", cv2.IMREAD_GRAYSCALE)
    print("ARI with other frame the same scene, different time, same view: ", ARI(gt_mask, pred_mask))
    print("fg-ARI with other frame the same scene, different time, same view: ", ARI(gt_mask, pred_mask, fg_mask))
    
    pred_mask = cv2.imread("../data/object_masks/Jumping/training_ 5.png", cv2.IMREAD_GRAYSCALE)
    print("ARI with other frame the same scene: ", ARI(gt_mask, pred_mask))
    print("fg-ARI with other frame the same scene: ", ARI(gt_mask, pred_mask, fg_mask))
   
    pred_mask = np.random.rand(gt_mask.shape[0], gt_mask.shape[1]) * 255.
    pred_mask = pred_mask.astype(int)
    print("ARI with random cluster: ", ARI(gt_mask, pred_mask))
    print("fg-ARI with random cluster: ", ARI(gt_mask, pred_mask, fg_mask))

    # validate LPIPS, PSNR, SSIM
    gt_img = torch.from_numpy(cv2.imread("../data/nvidia_data_full/Jumping/dense/mv_images/00000/cam01.jpg")[..., [2, 1, 0]]/ 255.*2. - 1.).cuda().permute(2, 0, 1)[None, ...]
    pred_img = torch.from_numpy(cv2.imread("../data/nvidia_data_full/Jumping/dense/mv_images/00000/cam01.jpg")[..., [2, 1, 0]]/ 255.*2. - 1.).cuda().permute(2, 0, 1)[None, ...]
    print("LPIPS with itself: ", loss_lpips(gt_img, pred_img))
    print("PSNR with itself: ", loss_psnr(gt_img, pred_img))
    print("SSIM with itself: ", loss_ssim(gt_img, pred_img))
    pred_img = torch.from_numpy(cv2.imread("../data/nvidia_data_full/Jumping/dense/mv_images/00000/cam06.jpg")[..., [2, 1, 0]]/ 255.*2. - 1.).cuda().permute(2, 0, 1)[None, ...]
    print("LPIPS with its spatial neighbor: ", loss_lpips(gt_img, pred_img))
    print("PSNR with its spatial neighbor: ", loss_psnr(gt_img, pred_img))
    print("SSIM with its spatial neighbor: ", loss_ssim(gt_img, pred_img))
    pred_img = torch.from_numpy(cv2.imread("../data/nvidia_data_full/Jumping/dense/mv_images/00001/cam01.jpg")[..., [2, 1, 0]]/ 255.*2. - 1.).cuda().permute(2, 0, 1)[None, ...]
    print("LPIPS with its temporal neighbor: ", loss_lpips(gt_img, pred_img))
    print("PSNR with its temporal neighbor: ", loss_psnr(gt_img, pred_img))
    print("SSIM with its temporal neighbor: ", loss_ssim(gt_img, pred_img))
    pred_img = torch.from_numpy(cv2.imread("../data/nvidia_data_full/Jumping/dense/mv_images/00000/cam08.jpg")[..., [2, 1, 0]]/ 255.*2. - 1.).cuda().permute(2, 0, 1)[None, ...]
    print("LPIPS with same time, different view: ", loss_lpips(gt_img, pred_img))
    print("PSNR with same time, different view: ", loss_psnr(gt_img, pred_img))
    print("SSIM with same time, different view: ", loss_ssim(gt_img, pred_img))
    pred_img = torch.from_numpy(cv2.imread("../data/nvidia_data_full/Jumping/dense/mv_images/00020/cam01.jpg")[..., [2, 1, 0]]/ 255.*2. - 1.).cuda().permute(2, 0, 1)[None, ...]
    print("LPIPS with same view, different time: ", loss_lpips(gt_img, pred_img))
    print("PSNR with same view, different time: ", loss_psnr(gt_img, pred_img))
    print("SSIM with same view, different time: ", loss_ssim(gt_img, pred_img))
    pred_img = torch.from_numpy(cv2.imread("../data/nvidia_data_full/Jumping/dense/mv_images/00015/cam09.jpg")[..., [2, 1, 0]]/ 255.*2. - 1.).cuda().permute(2, 0, 1)[None, ...]
    print("LPIPS with the same scene other view: ", loss_lpips(gt_img, pred_img))
    print("PSNR with the same scene other view: ", loss_psnr(gt_img, pred_img))
    print("SSIM with the same scene other view: ", loss_ssim(gt_img, pred_img))
    pred_img = (torch.rand(1, 3, pred_img.shape[2], pred_img.shape[3])*2. - 1.).cuda().double()
    print("LPIPS with random image: ", loss_lpips(gt_img, pred_img))
    print("PSNR with random image: ", loss_psnr(gt_img, pred_img))
    print("SSIM with random image: ", loss_ssim(gt_img, pred_img))