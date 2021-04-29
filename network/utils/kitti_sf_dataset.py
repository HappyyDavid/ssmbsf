import cv2
import torch
import os

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    '''
    wrapper of PIL.Image.open() regardless of warnings
    :param path: image path
    :return: PIL.Image object
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_flow_path(filename_path, dataset_path):
    '''
    concatenate data path
    :param filename_path:
    :param dataset_path:
    :return: {}, []
    '''
    filename = open(filename_path)
    data_path = {'left_up': [], 'left_down': []}
    flow_path = []

    for line in filename:

        line_split = line.split()
        # replace .jpg with .png
        line_split = list(map(lambda x: x.replace('jpg', 'png'), line_split))
        line_split = list(map(lambda x: dataset_path + x, line_split))
        # image02/former, image03/former, image02/latter, image03/latter
        data_path['left_up'].append(line_split[0])
        data_path['left_down'].append(line_split[1])
        flow_path.append(line_split[2])

    return data_path, flow_path

class KittiRandomEightPoints(Dataset):

    def __init__(self):

        self.data_path, self.flow_path = get_flow_path("/home2/Dave/pytorch/athena/utils/filenames/kitti_flow_val_files_200.txt",
                                                       "/home2/Dave/downloads/kitti_scene_flow_2015/")
        self.interp = Image.ANTIALIAS

    def __getitem__(self, index):

        data = {}

        # list: [l_u, l_d, r_u, r_d]
        ori_imgs_path = [
             self.data_path['left_up'][index],
             self.data_path['left_down'][index],
        ]

        ori_imgs = list(map(lambda x: pil_loader(x), ori_imgs_path))

        data[('ori', -1, 'lu')] = ori_imgs[0]
        data[('ori', -1, 'ld')] = ori_imgs[1]

        # copy from bridge
        # read flow from flo.png
        if self.flow_path is not None:
            flow = self.flow_path[index]
            # take obj_map
            t_ = flow.split('/')
            t_[-2] = 'obj_map'
            t_[0] = '/'
            obj_map = os.path.join(*t_)
            obj_map = cv2.imread(obj_map, -1)


            flow_image = cv2.imread(flow, -1)
            h, w, _ = flow_image.shape
            flo_img = flow_image[:,:,2:0:-1].astype(np.float32)
            invalid = (flow_image[:,:,0] == 0)

            flo_img = (flo_img - 32768) / 64
            flo_img[np.abs(flo_img) < 1e-10] = 1e-10
            flo_img[invalid, :] = 0

            f = torch.from_numpy(flo_img.transpose((2,0,1)))

        mask = (flow_image[:, :, 0] == 1)
        # find non-object resion
        mask_static = (obj_map == 0)
        mask = mask & mask_static
        mask = torch.from_numpy(mask.astype(np.float32)).type(torch.FloatTensor)
        # find non-zero flow points
        val_points = torch.nonzero(mask)
        # 32 * 8, random permute valid points, truncate the first 32 * 8.
        rand_samples = torch.randperm(val_points.shape[0], device=torch.device('cpu'))[:512*8].reshape(512, -1)



        sample_pts1 = val_points[rand_samples]
        sample_flow = f[:, sample_pts1[:, :, 0], sample_pts1[:, :, 1]].permute(1, 2, 0)
        sample_pts1 = sample_pts1.type(torch.FloatTensor)
        sample_pts2 = sample_pts1 + sample_flow
        pts = torch.cat([sample_pts1, sample_pts2], dim=2)
        return pts


    def __len__(self):
        return len(self.data_path['left_up'])

if __name__ == '__main__':


    dataset = KittiRandomEightPoints()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    for i, data in enumerate(train_dataset):
        print(i, data.shape)

    for i, data in enumerate(test_dataset):
        print(i, data.shape)