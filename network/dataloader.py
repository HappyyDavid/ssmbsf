import random
import torch
import cv2

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_data_path(filename_path, dataset_path):
    '''
    concatenate data path
    :param filename_path:
    :param dataset_path:
    :return: {}
    '''

    filename = open(filename_path)
    data_path = {'left_up': [], 'left_down': [], 'right_up': [], 'right_down': []}

    for line in filename:

        line_split = line.split()
        # replace .jpg with .png
        line_split = list(map(lambda x: x.replace('jpg', 'png'), line_split))
        line_split = list(map(lambda x: dataset_path + x, line_split))
        # image02/former, image03/former, image02/latter, image03/latter
        data_path['left_up'].append(line_split[0])
        data_path['left_down'].append(line_split[2])
        data_path['right_up'].append(line_split[1])
        data_path['right_down'].append(line_split[3])

    return data_path


class KittiCycleDataset(Dataset):

    def __init__(self, data_path, size, is_train, scales):

        self.data_path = data_path
        self.size = size
        self.is_train = is_train
        self.scales = scales
        self.interp = Image.ANTIALIAS
        # initialize colorjitter
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.to_tensor = transforms.ToTensor()
        # initialize resizer
        self.resize = {}
        for i in range(self.scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.size['h'] // s, self.size['w'] // s), interpolation=self.interp)


    def __getitem__(self, index):

        data = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # list: [l_u, l_d, r_u, r_d]
        ori_imgs_path = [
             self.data_path['left_up'][index],
             self.data_path['left_down'][index],
             self.data_path['right_up'][index],
             self.data_path['right_down'][index]
        ]
        ori_imgs = list(map(lambda x: pil_loader(x), ori_imgs_path))

        if do_flip:
            ori_imgs = list(map(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT), ori_imgs))

        # map: {'ori/aug', 0~3, 'lu/ld/ru/rd'}
        data[('ori', -1, 'lu')] = ori_imgs[0]
        data[('ori', -1, 'ld')] = ori_imgs[1]
        data[('ori', -1, 'ru')] = ori_imgs[2]
        data[('ori', -1, 'rd')] = ori_imgs[3]

        # augmentation
        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        for k in list(data):
            #img = data[k]
            if 'ori' in k:
                k_a, k_b, k_c = k
                for i in range(self.scales):
                    # TODO why? resize in last resized image
                    data[(k_a, i, k_c)] = self.resize[i](data[(k_a, i-1, k_c)])

        for k in list(data):
            img = data[k]
            if 'ori' in k:
                k_a, k_b, k_c = k
                data[(k_a, k_b, k_c)] = self.to_tensor(img)
                data[('aug', k_b, k_c)] = self.to_tensor(color_aug(img))
        # delete original image, because of different sizes
        for k_c in ['lu', 'ld', 'ru', 'rd']:
            del data[('ori', -1, k_c)]
            del data[('aug', -1, k_c)]
            # optional, delete original images
            for i in range(self.scales):
                del data[('ori', i, k_c)]

        return data

    def __len__(self):
        return len(self.data_path['left_up'])


def get_test_disp_data_path(filename_path, dataset_path):
    '''
    concatenate data path
    :param filename_path:
    :param dataset_path:
    :return: {}
    '''
    filename = open(filename_path)
    data_path = {'left_up': [], 'right_up': []}

    for line in filename:

        line_split = line.split()
        # replace .jpg with .png
        line_split = list(map(lambda x: x.replace('jpg', 'png'), line_split))
        line_split = list(map(lambda x: dataset_path + x, line_split))
        # image02/former, image03/former, image02/latter, image03/latter
        data_path['left_up'].append(line_split[0])
        data_path['right_up'].append(line_split[1])

    return data_path

def get_test_flow_data_path(filename_path, dataset_path):
    '''
    concatenate data path
    :param filename_path:
    :param dataset_path:
    :return: {}
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



class EvaluateDataset(Dataset):

    def __init__(self, data_path, size):

        self.data_path = data_path
        self.interp = Image.ANTIALIAS
        self.resizer  = transforms.Resize((size['h'], size['w']), interpolation=self.interp)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        data = {}
        # left_down = left_up, right_down = right_up
        # list: [l_u, l_d, r_u, r_d]
        ori_imgs_path = [
             self.data_path['left_up'][index],
             self.data_path['left_up'][index],
             self.data_path['right_up'][index],
             self.data_path['right_up'][index]
        ]

        #test optimize code, switch to predict disp2
        #ori_imgs_path = list(map(lambda x: x.replace('_10', '_11'), ori_imgs_path))

        ori_imgs = list(map(lambda x: pil_loader(x), ori_imgs_path))

        data[('ori', -1, 'lu')] = ori_imgs[0]
        data[('ori', -1, 'ld')] = ori_imgs[1]
        data[('ori', -1, 'ru')] = ori_imgs[2]
        data[('ori', -1, 'rd')] = ori_imgs[3]

        for k in list(data):
            k_a, k_b, k_c = k
            v = self.to_tensor(self.resizer(data[k]))
            data[('aug', 0, k_c)] = v

        for k in list(data):
            if 'ori' in k:
                del data[k]


        return data

    def __len__(self):
        return len(self.data_path['left_up'])


class EvaluateFlowDataset(Dataset):

    def __init__(self, data_path, flow_path, size):

        self.data_path = data_path
        self.flow_path = flow_path
        self.interp = Image.ANTIALIAS
        self.resizer  = transforms.Resize((size['h'], size['w']), interpolation=self.interp)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):

        data = {}

        # list: [l_u, l_d, r_u, r_d]
        ori_imgs_path = [
             self.data_path['left_up'][index],
             self.data_path['left_down'][index],
             self.data_path['left_up'][index],
             self.data_path['left_down'][index]
        ]
        ori_imgs = list(map(lambda x: pil_loader(x), ori_imgs_path))

        data[('ori', -1, 'lu')] = ori_imgs[0]
        data[('ori', -1, 'ld')] = ori_imgs[1]
        data[('ori', -1, 'ru')] = ori_imgs[2]
        data[('ori', -1, 'rd')] = ori_imgs[3]

        # used for upsample flow
        img_w, img_h = data[('ori', -1, 'lu')].size

        for k in list(data):
            k_a, k_b, k_c = k
            v = self.to_tensor(self.resizer(data[k]))
            data[('aug', 0, k_c)] = v

        for k in list(data):
            if 'ori' in k:
                del data[k]

        # copy from bridge
        if self.flow_path is not None:
            flow = self.flow_path[index]
            flow_image = cv2.imread(flow, -1)
            h, w, _ = flow_image.shape
            flo_img = flow_image[:,:,2:0:-1].astype(np.float32)
            invalid = (flow_image[:,:,0] == 0)

            flo_img = (flo_img - 32768) / 64
            flo_img[np.abs(flo_img) < 1e-10] = 1e-10
            flo_img[invalid, :] = 0

            f = torch.from_numpy(flo_img.transpose((2,0,1)))
            mask = torch.from_numpy((flow_image[:,:,0] == 1).astype(np.float32)).type(torch.FloatTensor)

            flow_data = {
                "flo": f.type(torch.FloatTensor), "mask": mask,
                "h": h, "w": w,
                "img_h": img_h, "img_w": img_w
            }

            return data, flow_data

        return data


    def __len__(self):
        return len(self.data_path['left_up'])





'''
# test code

from athena_config import Config
from torch.utils.data import DataLoader
args = Config()
data_path = get_data_path(args.filename_path, args.dataset_path)
dataset = KittiCycleDataset(data_path, {'h': 192, 'w': 640}, True, 4)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1, drop_last=True)
from tensorboardX import SummaryWriter
writer = SummaryWriter()

for idx, data in enumerate(dataloader):
    writer.add_image('ori_img/scale_0', data[('ori', 0, 'lu')][0], idx)
    writer.add_image('ori_img/scale_1', data[('ori', 1, 'lu')][0], idx)
    writer.add_image('ori_img/scale_2', data[('ori', 2, 'lu')][0], idx)
    writer.add_image('ori)img/scale_3', data[('ori', 3, 'lu')][0], idx)

    writer.add_image('aug_img/scale_0', data[('aug', 0, 'lu')][0], idx)
    writer.add_image('aug_img/scale_1', data[('aug', 1, 'lu')][0], idx)
    writer.add_image('aug_img/scale_2', data[('aug', 2, 'lu')][0], idx)
    writer.add_image('aug_img/scale_3', data[('aug', 3, 'lu')][0], idx)
    print(idx)

'''