import torch
from athena_layers import *

def set_train(models):
    """Convert all models to training mode
    """
    for m in models.values():
        m.train()

def set_eval(models):
    """Convert all models to testing/evaluation mode
    """
    for m in models.values():
        m.eval()


def compute_avg_reprojection_loss(pred, target, mask=None):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = SSIM().cuda()(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    if mask == None:
        return reprojection_loss.mean()
    else:
        return (reprojection_loss * mask).mean() / mask.mean()



def compute_mini_reprojection_loss(pred, target):
    """Computes minimum reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = SSIM().cuda()(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    to_optim, idx = reprojection_loss.min(dim=0)

    return to_optim.mean()

def image_sequence(data, mode=None):
    '''
    concatenate former, latter image sequence, [6/4 * B, C, H, W]
    :param data:
    :param mode: stereo, flow, cross
    :return:
    '''
    former = None
    latter = None
    assert (mode=='stereo-flow-cross') or (mode=='stereo-flow') or (mode=='stereo') or (mode=='flow')

    if mode == 'stereo-flow-cross':
        former = torch.cat((data[('aug', 0, 'lu')], data[('aug', 0, 'ld')],
                            data[('aug', 0, 'lu')], data[('aug', 0, 'ru')],
                            data[('aug', 0, 'lu')], data[('aug', 0, 'ru')]), dim=0)

        latter = torch.cat((data[('aug', 0, 'ru')], data[('aug', 0, 'rd')],
                            data[('aug', 0, 'ld')], data[('aug', 0, 'rd')],
                            data[('aug', 0, 'rd')], data[('aug', 0, 'ld')]), dim=0)

    elif mode == 'stereo-flow':
        # bridge implement image sequence order
        # former = torch.cat((data[('aug', 0, 'lu')], data[('aug', 0, 'lu')], data[('aug', 0, 'ru')], data[('aug', 0, 'lu')]), dim=0)
        # latter = torch.cat((data[('aug', 0, 'ru')], data[('aug', 0, 'rd')], data[('aug', 0, 'rd')], data[('aug', 0, 'ru')]), dim=0)
        former = torch.cat((data[('aug', 0, 'lu')], data[('aug', 0, 'ld')],
                            data[('aug', 0, 'lu')], data[('aug', 0, 'ru')]), dim=0)

        latter = torch.cat((data[('aug', 0, 'ru')], data[('aug', 0, 'rd')],
                            data[('aug', 0, 'ld')], data[('aug', 0, 'rd')]), dim=0)

    elif mode == 'stereo':
        former = torch.cat((data[('aug', 0, 'lu')], data[('aug', 0, 'ld')]), dim=0)
        latter = torch.cat((data[('aug', 0, 'ru')], data[('aug', 0, 'rd')]), dim=0)

    elif mode == 'flow':
        former = torch.cat((data[('aug', 0, 'lu')], data[('aug', 0, 'ru')]), dim=0)
        latter = torch.cat((data[('aug', 0, 'ld')], data[('aug', 0, 'rd')]), dim=0)

    return former, latter

def image_sequences_(data, mode, n_scales):
    '''
    concatenate former, latter image sequence in different scales
    :param data:
    :param mode: stereo, flow, cross
    :return:
    '''
    formers = {}
    latters = {}
    assert (mode=='stereo-flow-cross') or (mode=='stereo-flow') or (mode=='stereo') or (mode=='flow')

    if mode == 'stereo-flow-cross':
        for i in range(n_scales):

            former_ = torch.cat((data[('aug', i, 'lu')], data[('aug', i, 'ld')],
                                 data[('aug', i, 'lu')], data[('aug', i, 'ru')],
                                 data[('aug', i, 'lu')], data[('aug', i, 'ru')]), dim=0)

            latter_ = torch.cat((data[('aug', i, 'ru')], data[('aug', i, 'rd')],
                                 data[('aug', i, 'ld')], data[('aug', i, 'rd')],
                                 data[('aug', i, 'rd')], data[('aug', i, 'ld')]), dim=0)

            formers[('former', i)] = former_
            latters[('latter', i)] = latter_

    return formers, latters

def model_inputs(former, latter):
    '''
    concatenate forward, backward model inputs, [6*B, C*2, H, W]
    :param former:
    :param latter:
    :return:
    '''
    forward_in = torch.cat((former, latter), dim=1)
    bacward_in = torch.cat((latter, former), dim=1)

    return forward_in, bacward_in

# second order smoothness loss in 'bridge' code
def cal_grad2_error(flo, image, beta=10):
    """
    Calculate the image-edge-aware second-order smoothness loss for flo
    """

    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    img_dx, img_dy = gradient(image)
    w_x = torch.exp(-10.0 * torch.mean(torch.abs(img_dx), 1, keepdim=True))
    w_y = torch.exp(-10.0 * torch.mean(torch.abs(img_dy), 1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, _ = gradient(dx)
    _, dy2 = gradient(dy)

    '''
    return (torch.mean(beta * w_x[:, :, :, 1:] * torch.abs(dx2)) + torch.mean(
        beta * w_y[:, :, 1:, :] * torch.abs(dy2))) / 2.0
    '''
    return (torch.mean(w_x[:, :, :, 1:] * torch.abs(dx2)) + torch.mean(w_y[:, :, 1:, :] * torch.abs(dy2))) / 2.0



def compute_smooth_loss(disp, img):
    """
    Calculate the image-edge-aware first-order smoothness loss for flo
    """
    grad_disp_x = torch.mean(torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :]), 3, keepdim=True)
    grad_disp_y = torch.mean(torch.abs(disp[:, :-1, :, :] - disp[:, 1:, :, :]), 3, keepdim=True)

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True).permute([0, 2, 3, 1])
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True).permute([0, 2, 3, 1])

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def compute_lr_loss(disp1, disp2):
    """
        compute left-right consistency loss
    :param disp1:
    :param disp2:
    :return: loss
    """
    # TODO: define disp2 is minus one better
    lr_loss = torch.mean(torch.abs(disp1 + disp2))
    return lr_loss

def compute_fw_loss(flo_diff, mask=None):
    """
        compute forward-backward flow consistency loss
    :param flo_diff: difference of flow and warped-flow
    :param mask: disocclusion mask
    :return: loss
    """
    if mask == None:
        return torch.mean(torch.abs(flo_diff))
    else:
        return torch.abs(flo_diff * mask).mean() / mask.mean()


def scale_disps(forward_out, backward_out, scales, size):
    '''
        scale disparities at different size to the same one [H, W].
    :param forward_out: forward disparities
    :param backward_out: backward disparities
    :param scales: number of scale
    :param size: target size
    :return: None
    '''
    for i in range(scales):

        # forward part
        f_disp = forward_out[("disp", i)]
        f_disp = F.interpolate(f_disp, [size['h'], size['w']], mode="bilinear", align_corners=False)
        forward_out[("disp_scale", i)] = f_disp.permute([0, 2, 3, 1]) * (2 ** i)  # [B, H, W, 2]
        # backward part
        b_disp = backward_out[("disp", i)]
        b_disp = F.interpolate(b_disp, [size['h'], size['w']], mode="bilinear", align_corners=False)
        backward_out[("disp_scale", i)] = b_disp.permute([0, 2, 3, 1]) * (2 ** i)  # [B, H, W, 2]


def get_grid(size):
    """
    generate normalized pixel coordinates with [b, h, w, 2]
    :param size: tensor size
    :return: normalized pixel coordinates
    """
    b, h, w, c = size
    hori = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(b, -1, h, -1)
    vert = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(b, -1, -1, w)
    grid = torch.cat([hori, vert], dim=1).permute([0, 2, 3, 1])
    return grid

def get_grids(size, n_scales):
    """
    generate normalized pixel coordinates in different levels
    :param size: tensor size
    :return: normalized pixel coordinates
    """
    grids = {}
    b, h, w, _ = size
    for i in range(n_scales):
        b_ = b
        h_ = int(h / 2**i)
        w_ = int(w / 2**i)
        hori_ = torch.linspace(-1.0, 1.0, w_).view(1, 1, 1, w_).expand(b_, -1, h_, -1)
        vert_ = torch.linspace(-1.0, 1.0, h_).view(1, 1, h_, 1).expand(b_, -1, -1, w_)
        grid_ = torch.cat([hori_, vert_], dim=1).permute([0, 2, 3, 1]).cuda()
        grids[('grid', i)] = grid_

    return grids

def length_sq(x):
    return torch.sum(x**2, 1, keepdim=True) # flo_x^2 + flo_y^2


def get_flow_mask(forward_flo, backward_flo, grid):

    # consistent with <bidirectional census, AAAI 2018>
    # reshape flow from [B, H, W, 2] to [B, 2, H, W]
    f_flo = forward_flo.permute(0, 3, 1, 2)
    b_flo = backward_flo.permute(0, 3, 1, 2)

    warp_f_flo = F.grid_sample(b_flo, forward_flo + grid, padding_mode="border")
    warp_b_flo = F.grid_sample(f_flo, backward_flo + grid, padding_mode="border")

    f_flo_diff = warp_f_flo + f_flo
    b_flo_diff = warp_b_flo + f_flo

    f_length_sq = length_sq(f_flo) + length_sq(warp_f_flo)
    b_length_sq = length_sq(b_flo) + length_sq(warp_b_flo)

    f_occ_thre = 0.01 * f_length_sq + 0.5
    b_occ_thre = 0.01 * b_length_sq + 0.5

    f_occ = (length_sq(f_flo_diff) > f_occ_thre).type(torch.cuda.FloatTensor)
    b_occ = (length_sq(b_flo_diff) > b_occ_thre).type(torch.cuda.FloatTensor)

    return 1 - f_occ, 1 - b_occ, f_flo_diff, b_flo_diff


# TODO can i use the minimum reconstruction loss?
#   compute_avg_reprojection_loss -> compute_min_reprrojection_loss
#   nope, it's seem to make training quickly converge

grids = None

def compute_loss(forward_out, backward_out, formers, latters, scales):

    # TODO bad implement, grids is global variable
    # here only instance the grids once.
    global grids
    if grids == None:
        grids = get_grids(forward_out[("disp_scale", 0)].shape, 4)

    losses = {}
    total_loss = 0

    for i in range(scales):
        # image reconstruction loss, gradient smoothness loss.

        # compute loss in scale disp
        former, latter = formers[('former', 0)], latters[('latter', 0)]
        f_disp = forward_out[("disp_scale", i)]
        b_disp = backward_out[("disp_scale", i)]
        grid = grids[('grid', 0)]
        # forward part
        warped_former = F.grid_sample(latter, f_disp + grid, padding_mode="border")
        forward_out[('warp', i)] = warped_former
        f_rec_img_loss = compute_avg_reprojection_loss(warped_former, former)
        f_grad2_loss = compute_smooth_loss(f_disp, former) / (i + 1)# (2 ** i)
        # backward part
        warped_latter = F.grid_sample(former, b_disp + grid, padding_mode="border")
        backward_out[('warp', i)] = warped_latter
        b_rec_img_loss = compute_avg_reprojection_loss(warped_latter, latter)
        b_grad2_loss = compute_smooth_loss(b_disp, latter) / (i + 1)#(2 ** i)

        # ***************************- TEST -*************************** #
        # compute loss in original disp

        former, latter = formers[('former', i)], latters[('latter', i)]
        f_disp = forward_out[("disp", i)].permute(0, 2, 3, 1)
        b_disp = backward_out[("disp", i)].permute(0, 2, 3, 1)
        grid = grids[('grid', i)]
        # forward part
        warped_former_ = F.grid_sample(latter, f_disp + grid, padding_mode="border")
        forward_out[('warp_', i)] = warped_former_
        f_rec_img_loss_ = compute_avg_reprojection_loss(warped_former_, former)
        f_grad2_loss_ = compute_smooth_loss(f_disp, former) / (2 ** i)
        # backward part
        warped_latter_ = F.grid_sample(former, b_disp + grid, padding_mode="border")
        backward_out[('warp_', i)] = warped_latter_
        b_rec_img_loss_ = compute_avg_reprojection_loss(warped_latter_, latter)
        b_grad2_loss_ = compute_smooth_loss(b_disp, latter) / (2 ** i)

        # ***************************- TEST -*************************** #

        # ***************************- TEST -*************************** #
        # bidirectional census loss
        # WARNING only implement in image sequence mode is 'flow'
        '''
        f_no_occ, b_no_occ, f_flo_diff, b_flo_diff = get_flow_mask(f_disp, b_disp, grid)
        if epoch < 10:
            f_no_occ, b_no_occ = None, None
        '''
        '''
        f2b_fw_loss = compute_fw_loss(f_flo_diff, f_no_occ)
        b2f_fw_loss = compute_fw_loss(b_flo_diff, b_no_occ)
        '''
        # ***************************- TEST -*************************** #

        # ***************************- TEST -*************************** #
        # left-right consistency
        # WARNING only implement in image sequence mode is 'stereo'
        '''
        # forward part
        warped_f_disp = F.grid_sample(-b_disp.permute(0, 3, 1, 2), f_disp + grid, padding_mode="border")
        f2b_lr_loss = compute_lr_loss(warped_f_disp.permute(0, 2, 3, 1), f_disp)
        # backward part
        warped_b_disp = F.grid_sample(-f_disp.permute(0, 3, 1, 2), b_disp + grid, padding_mode="border")
        b2f_lr_loss = compute_lr_loss(warped_b_disp.permute(0, 2, 3, 1), b_disp)
        '''
        # ***************************- TEST -*************************** #

        # ***************************- TEST -*************************** #
        # vertical flow zero loss
        # add a supervision on vertical flow should be zero n disparity estimation setting.
        # the first 2 * batch_size channel in forward_out is disparity prediction.
        '''
        disp_ver_flo = forward_out[("disp_scale", i)][0:4, ..., 1]
        disp_ver_flo_loss = disp_ver_flo.mean()
        losses[(i, 'dvf')] = disp_ver_flo_loss
        total_loss += disp_ver_flo_loss * 0.1
        '''
        # ***************************- TEST -*************************** #

        # record loss in different level
        losses[(i, 'rec')] = (f_rec_img_loss + b_rec_img_loss) / 2
        losses[(i, 'grd')] = 20 * (f_grad2_loss + b_grad2_loss) / 2
        #losses[(i, 'lr')]  = 0.5 * (f2b_lr_loss + b2f_lr_loss) / 2
        #losses[(i, 'fw')]  = 0.5 * (f2b_fw_loss + b2f_fw_loss) / 2
        losses[(i, 'tot')] = losses[(i, 'rec')] + losses[(i, 'grd')]
        # sum of all loss
        total_loss += losses[(i, 'tot')]

    total_loss = total_loss / scales
    losses['total'] = total_loss

    return losses