import torch

'''
b, c, h, w = 12, 2, 192, 640

tensorFlow = torch.randn([b, c, h, w])
tensorHorizaontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
tensorGrid = torch.cat([tensorHorizaontal, tensorVertical], dim=1).permute([0, 2, 3, 1])

Hori = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, -1, h, -1)
Vert = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, -1, -1, w)
Grid = torch.cat([Hori, Vert], dim=1).permute([0, 2, 3, 1])


'''

'''
from athena_config import Config
C = Config()

import os
ckp_dir = os.path.join(C.checkpoint_path, C.checkpoint_name)
if not os.path.exists(ckp_dir):
    os.makedirs(ckp_dir)

'''

'''
import numpy as np
idx = 0
gt_depth = np.load('idon.npy', allow_pickle=True)
for i in range(len(gt_depth)):
    if (375, 1242) != gt_depth[i].shape:
        idx = idx + 1
        print(gt_depth[i].shape)

print(idx)

'''


'''
# test save ndarray with .mat
import scipy.io as io

flow_op = io.loadmat('000000_10_flo.mat')['flow_12l_op']

print('haha')
'''

# caculate mask of disp and warped disp
import torch
import numpy as np
import scipy.io as io
import torch.nn.functional as F

data_path = "/home2/Dave/pytorch/athena/t_optim/"

for i in range(200):
    disp1 = np.load(data_path + '{}_disp1.npy'.format(i))
    disp2 = np.load(data_path + '{}_disp2.npy'.format(i))
    flo = np.load(data_path + '{}_flo.npy'.format(i))

    disp1 = torch.from_numpy(disp1).unsqueeze(0).unsqueeze(0)
    disp2 = torch.from_numpy(disp2).unsqueeze(0).unsqueeze(0)
    flo = torch.from_numpy(flo).permute(0, 2, 3, 1)

    warp_disp1 = F.grid_sample(disp2, flo, padding_mode='reflection')
    inlier = torch.abs(disp1 -  warp_disp1) < 30
    inlier = inlier.squeeze()


    save_name = str(i).zfill(6) + '_wap_mask.mat'
    io.savemat(save_name, {'in_mask': inlier.numpy()})

    print(save_name)