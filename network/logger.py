import torch

from tensorboardX import SummaryWriter

from athena_config import Config
from tfoptflow_optflow import *

writer = SummaryWriter(comment=Config().logger_suffix)

def logger_loss(losses, idx, scales):

    for i in range(scales):
        # logger image reconstruction, gradient smoothness, total in different level
        for k_b in ['rec', 'grd', 'tot']:
            # if k_b in losses.keys()
            writer.add_scalar('scale_{}/{}'.format(i, k_b), losses[(i, k_b)], idx)

    writer.add_scalar('total', losses['total'], idx)

def logger_flo(img, idx):
    writer.add_image('flo', img, idx)

def logger_img(string, img, idx):
    writer.add_image(string, img, idx)

def logger_scalar(string, value, idx):
    writer.add_scalar(string, value, idx)


def tbx_logger(C, index, data, models, forward_out, backward_out, losses):

    h, w = C.input_height, C.input_width
    # idx 0 is stereo
    scale_flo = forward_out[('disp', 0)][0].permute([1, 2, 0]).detach().cpu()
    # how to recover absolute flow from normalized one?
    # recover flow, copied by spynet-pytorch
    scale_flo[..., 0] = scale_flo[..., 0] * (w / 2)
    scale_flo[..., 1] = scale_flo[..., 1] * (h / 2)

    flo_img = flow_to_img(scale_flo.numpy())
    flo_img = torch.from_numpy(flo_img).permute([2, 0, 1])
    logger_flo(flo_img, index)

    # logger original image/ warped image
    for i in range(4):
        logger_img('img_{}/wrp_lu'.format(i), forward_out[('warp', i)][0], index)
        logger_img('img_{}/wrp_lu_'.format(i), forward_out[('warp_', i)][0], index)
        logger_img('img_{}/ori_lu'.format(i), data[('aug', i, 'lu')][0], index)
        logger_img('img_{}/ori_ru'.format(i), data[('aug', i, 'ru')][0], index)

    logger_scalar('conv1/wei', models["encoder"].encoder.conv1.weight[0, 0, 0, 0], index)
    logger_scalar('conv1/grd', models["encoder"].encoder.conv1.weight.grad[0, 0, 0, 0], index)


    logger_loss(losses, index, C.num_scales)

