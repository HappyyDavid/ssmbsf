import torch
import torch.nn as nn
import torch.nn.functional as F

class Corr1d(nn.Module):

    def __init__(self,max_disp=10):

        super(Corr1d, self).__init__()
        self.max_disp = max_disp

    def forward(self, x, y):

        corr_tensors = []
        for i in range(-self.max_disp, 0, 1):
            s1 = y.narrow(3, 0, y.shape[3] + i)
            shifted = F.pad(s1, (-i, 0, 0, 0), "constant", 0.0)
            corr = torch.mean(shifted * x, 1)
            corr_tensors.append(corr)

        for i in range(self.max_disp + 1):
            s2 = x.narrow(3, i, x.shape[3] - i)
            shifted = F.pad(s2, (0, i, 0, 0), "constant", 0.0)
            corr = torch.mean(shifted * y, 1)
            corr_tensors.append(corr)

        temp = torch.stack(corr_tensors)
        out = torch.transpose(temp, 0, 1)
        return torch.mean(out, 1).squeeze()


class CostVolume(nn.Module):

    def __init__(self, h_s, w_s):
        super(CostVolume, self).__init__()
        assert h_s % 2 == 0 and w_s % 2 == 0
        self.h_s = h_s
        self.w_s = w_s

    def forward(self, x1, x2):

        p2d = (self.h_s, self.h_s, self.w_s, self.w_s)
        padded_x2 = F.pad(x2, p2d, "constant", 0.0)
        h_max_offset = self.h_s * 2 + 1
        w_max_offset = self.w_s * 2 + 1
        _, _, h, w = x1.shape

        cost_vol = []
        for y in range(0, h_max_offset):
            for x in range(0, w_max_offset):
                slice = padded_x2[:, :, y:y+h, x:x+w]
                cost = torch.mean(x1*slice, dim=1, keepdim=True)
                cost_vol.append(cost)
        cost_vol = torch.cat(cost_vol, dim=1)
        cost_vol = F.leaky_relu(cost_vol, 0.1)

        return cost_vol



import time

import numpy as np
from athena_resnet_encoder import ResnetEncoder
from PWCNet import conv, predict_flow, deconv

x = torch.rand(8, 3, 640, 192).cuda()
y = torch.rand(8, 3, 640, 192).cuda()

res18_encoder = ResnetEncoder(18, True, 1).cuda()
x_features = res18_encoder(x)
y_features = res18_encoder(y)

corr2d = CostVolume(4, 4).cuda()
loss = 0.0

corr6 = corr2d(x_features[0], y_features[0])

nd = (2 * 4 + 1) ** 2
dd = np.cumsum([128, 128, 96, 64, 32])

od = nd
conv6_0 = conv(od, 128, kernel_size=3, stride=1).cuda()
conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1).cuda()
conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1).cuda()
conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1).cuda()
conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1).cuda()
predict_flow6 = predict_flow(od + dd[4]).cuda()
deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1).cuda()
upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1).cuda()

x = torch.cat((conv6_0(corr6), corr6), 1)
x = torch.cat((conv6_1(x), x), 1)
x = torch.cat((conv6_2(x), x), 1)
x = torch.cat((conv6_3(x), x), 1)
x = torch.cat((conv6_4(x), x), 1)
flow6 = predict_flow6(x)
up_flow6 = deconv6(flow6)
up_feat6 = upfeat6(x)



'''
from correlation_package.correlation import Correlation
corr    = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1).cuda()
s_time = time.time()
out = corr2d(x, y)
e_time = time.time()
print(e_time-s_time, 'haha')
'''

