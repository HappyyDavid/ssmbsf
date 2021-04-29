# something must be wrong when i recover the depth of frame2

import cv2
import scipy.io as io

import numpy as np

def read_flow(path):
    flow_image = cv2.imread(path, -1)
    h, w, _ = flow_image.shape
    flo_img = flow_image[:, :, 2:0:-1].astype(np.float32) # [h, w, 2]
    invalid = (flow_image[:, :, 0] == 0)

    flo_img = (flo_img - 32768) / 64
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] = 0

    return flo_img.transpose(2, 0, 1), 1- invalid # [2, h, w]



kitti_sf_2015_dataset_path = '/home2/Dave/downloads/kitti_scene_flow_2015'
sample = str(0).zfill(6) + "_10.png"

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
baseline = 0.53272545




obj_1 = cv2.imread(kitti_sf_2015_dataset_path + "/training/obj_map/" + sample, -1)
mask_static = (obj_1 == 0)
mask_dynamic = (obj_1 == 1)

#generate ground truth depth of left camera frame1, frame2
disp_1 = cv2.imread(kitti_sf_2015_dataset_path + "/training/disp_occ_0/" + sample, -1)
disp_1 = disp_1.astype(np.float32) / 256
h, w = disp_1.shape
val_ratio = np.count_nonzero(disp_1) / (h * w) # avg_val_ratio = 0.1973185548524565
mask_1 = disp_1 > 0
depth_1 = width_to_focal[w] * 0.53272545 / (disp_1 + (1.0 - mask_1))

disp_2 = cv2.imread(kitti_sf_2015_dataset_path + "/training/disp_occ_1/" + sample, -1)
disp_2 = disp_2.astype(np.float32) / 256
h, w = disp_2.shape
mask_2 = disp_2 > 0
depth_2 = width_to_focal[w] * 0.53272545 / (disp_2 + (1.0 - mask_2))

noc_flow, noc_mask_f = read_flow(kitti_sf_2015_dataset_path + "/training/flow_noc/" + sample)
occ_flow, occ_mask_f = read_flow(kitti_sf_2015_dataset_path + "/training/flow_occ/" + sample)

# generate image coordinates
x = np.arange(1, w + 1) # [1, w]
y = np.arange(1, h + 1) # [1, h]
xv, yv = np.meshgrid(x, y) # [h, w]
pixel_coor = np.stack([xv, yv]) # [2, h, w]

img_corr_1to2 = pixel_coor + noc_flow
inlier_img_corr_1to2 = (img_corr_1to2[0, :, :] > 0) & (img_corr_1to2[0, :, :] < w) & \
                       (img_corr_1to2[1, :, :] > 0) & (img_corr_1to2[1, :, :] < h) & \
                       (noc_mask_f != 0)

# find common region of mask_1, mask_f, mask_12
mask_12 = inlier_img_corr_1to2
mask = (mask_1 != 0) & (noc_mask_f != 0) & (mask_12 != 0) & \
       (mask_1 == noc_mask_f) & (noc_mask_f == mask_12) & (mask_1 == mask_12)
# static part
mask = mask & mask_dynamic

dpt1 = depth_1[mask]
dpt2 = depth_2[mask]
flo = np.floor(noc_flow[:, mask]) # TODO floor of flow
pix = pixel_coor[:, mask]

io.savemat(str(0).zfill(6) + "_10.mat", {'depth1l': dpt1, 'depth2l': dpt2, 'flow': flo, 'pixel': pix})

# mask_1 == mask_2
# kitti scene flow 2015 devkit readme.txt
# - disp_xxx_0: Disparity maps of first image pairs in reference frame
#               (only regions which don't leave the image domain).
#
# - disp_xxx_1: Disparity information of second image pair mapped into the
#               reference frame (first left image) via the true optical flow.

# find the common valid region of disp_0 and flow
