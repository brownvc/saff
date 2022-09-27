import open3d as o3d
import torch
import cv2
import numpy as np

'''look square because lives in ndc space!!!'''

filename = "logs/experiment_4_F00-30/render-slowmo_full_path_090001/poses/049.pt"


depths = -cv2.imread(filename.replace("poses", "depths").replace(".pt", ".jpg"), cv2.IMREAD_GRAYSCALE)[..., None]/255.
colors = cv2.imread(filename.replace("poses", "images").replace(".pt", ".jpg"))[..., [2, 1, 0]]/255.


rays = torch.load(filename)
rays_o = rays[0]
rays_d = rays[1]

depths = torch.from_numpy(depths).to(rays_o.device)

points = rays_o[...,None,:] + rays_d[...,None,:] * depths[...,:,None]
#assert False, torch.max(depths.shape)
#assert False, [colors.shape, points.shape]
xyz = points.reshape((-1, 3)).cpu().numpy()
#xyz = rays_o.view(-1, 3).cpu().numpy()
colors = colors.reshape((-1, 3))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("test.ply", pcd)
#assert False, [rays.shape, depth.shape, points.shape]

