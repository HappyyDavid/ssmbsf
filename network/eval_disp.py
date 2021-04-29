import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import scipy.io as io
import pandas as pd

from torch.utils.data import DataLoader

from athena_config import Config
C = Config()

from athena_dataloader import get_test_disp_data_path, EvaluateDataset
from athena_resnet_encoder import ResnetEncoder
from athena_depth_decoder import DepthDecoder
from athena_loss import image_sequence, model_inputs
from bridge_evaluation_utils import *
from athena_groundtruth import eigen_test_gt_save_path, eigen_test_gt_depth_save_path

def generate_prediction_disp(ckp_idx, input_mode, test_mode):

    n_pred_disps = None
    test_filename_pth = None
    dataset_pth = None

    if test_mode == 'eigen':
        n_pred_disps = 697
        test_filename_pth = C.test_stereo_eigen_filename_path
        dataset_pth = C.kitti_raw_dataset_path
    elif test_mode == 'kitti':
        n_pred_disps = 200
        test_filename_pth = C.test_stereo_kitti_filename_path
        dataset_pth = C.kitti_sf_2015_dataset_path

    # data prepare
    data_path = get_test_disp_data_path(test_filename_pth, dataset_pth)
    dataset = EvaluateDataset(data_path, {'h': C.input_height, 'w': C.input_width})
    dataloader = DataLoader(dataset, batch_size=C.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # load model
    ckp_load_dir = os.path.join(C.checkpoint_path, C.test_kitti_ckp_name, "weights_{}".format(ckp_idx))
    encoder_path = os.path.join(ckp_load_dir, "encoder.pth")
    decoder_path = os.path.join(ckp_load_dir, "decoder.pth")

    encoder = ResnetEncoder(C.num_layers, False, 2)
    decoder = DepthDecoder(encoder.num_ch_enc, C.scales, num_output_channels=2)

    encoder_dict = torch.load(encoder_path)
    h = encoder_dict['height']
    w = encoder_dict['width']
    del encoder_dict['height']
    del encoder_dict['width']

    encoder.load_state_dict({k: v for k, v in encoder_dict.items()})
    decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()

    # allocate space to save network's predictions
    pred_disps = np.zeros((n_pred_disps, C.input_height, C.input_width), dtype=np.float32)
    index = 0
    # generate predicts
    with torch.no_grad():

        for idx, data in enumerate(dataloader):

            #print(idx)
            for key, item in data.items():
                data[key] = item.cuda()

            former, latter = image_sequence(data, mode=input_mode)
            forward_in, backward_in = model_inputs(former, latter)
            forward_features = encoder(forward_in)
            forward_out = decoder(forward_features)

            disps = forward_out['disp', 0]
            b, _, _ , _ = disps.shape
            # b must be a even number, cause forward-backward
            for i in range(int(b/2)):
                pred_disps[index + i] = disps[i, 0].detach().cpu().numpy()
            index = index + int(b/2)

    # why should we times -1? the definition of disparity must be positive
    np.save('./test_disps.npy', -pred_disps)

def compute_eigen_disp_errors(pred_disps):
    n_eigen_samples = 697
    # load pre-computed ground truth
    focal_length = np.load(eigen_test_gt_save_path + 'focal_length.npy')
    baseline = np.load(eigen_test_gt_save_path + 'baseline.npy')
    img_size = np.load(eigen_test_gt_save_path + 'img_size.npy')

    # check ground truth completeness
    assert len(os.listdir(eigen_test_gt_depth_save_path)) == n_eigen_samples
    assert focal_length.shape == (n_eigen_samples, 1)
    assert baseline.shape == (n_eigen_samples, 1)
    assert img_size.shape == (n_eigen_samples, 2)
    '''
    for idx in range(n_eigen_samples):
        gt_depth = np.load(eigen_test_gt_depth_save_path + 'depth_{}.npy'.format(idx))
        assert (gt_depth.shape == img_size[idx]).all()
    
    '''
    # compute the estimated depth
    pred_depths = []
    for idx in range(n_eigen_samples):

        disp_pred = cv2.resize(pred_disps[idx], (img_size[idx][1], img_size[idx][0]), interpolation=cv2.INTER_LINEAR)
        # TODO is that scale right?
        disp_pred = disp_pred * disp_pred.shape[1] / 2
        # convert disparity to depth
        depth_pred = (baseline[idx] * focal_length[idx]) / disp_pred
        depth_pred[np.isinf(depth_pred)] = 0
        pred_depths.append(depth_pred)

    rms = np.zeros(n_eigen_samples, np.float32)
    log_rms = np.zeros(n_eigen_samples, np.float32)
    abs_rel = np.zeros(n_eigen_samples, np.float32)
    sq_rel = np.zeros(n_eigen_samples, np.float32)
    d1_all = np.zeros(n_eigen_samples, np.float32)
    a1 = np.zeros(n_eigen_samples, np.float32)
    a2 = np.zeros(n_eigen_samples, np.float32)
    a3 = np.zeros(n_eigen_samples, np.float32)

    # compute the error
    for i in range(n_eigen_samples):

        gt_depth = np.load(eigen_test_gt_depth_save_path + 'depth_{}.npy'.format(i))
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80] = 80

        mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

def compute_kitti_disp_errors(pred_disps):

    n_kitti_samples = 200
    gt_disps = load_gt_disp_kitti(C.kitti_sf_2015_dataset_path)
    gt_depths, pred_depths, pred_disps_resized = convert_disps_to_depths_kitti(gt_disps, pred_disps)

    rms     = np.zeros(n_kitti_samples, np.float32)
    log_rms = np.zeros(n_kitti_samples, np.float32)
    abs_rel = np.zeros(n_kitti_samples, np.float32)
    sq_rel  = np.zeros(n_kitti_samples, np.float32)
    d1_all  = np.zeros(n_kitti_samples, np.float32)
    a1      = np.zeros(n_kitti_samples, np.float32)
    a2      = np.zeros(n_kitti_samples, np.float32)
    a3      = np.zeros(n_kitti_samples, np.float32)


    # ***************************- TEST -*************************** #
    # save disp
    # for i in range(n_kitti_samples):
    #     np.save('{}_disp2.npy'.format(i), pred_disps_resized[i])
    # ***************************- TEST -*************************** #


    # ***************************- TEST -*************************** #
    # save depth
    # for i in range(n_kitti_samples):
    #     save_file_ = str(i).zfill(6) + '_2l.mat'
    #     save_npy_ = pred_depths[i]
    #     io.savemat(save_file_, {'depth2l': save_npy_})
    # ***************************- TEST -*************************** #



    for i in range(n_kitti_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80] = 80

        gt_disp = gt_disps[i]
        mask = gt_disp > 0 # different definition with test_mode = 'eigen'
        pred_disp = pred_disps_resized[i]

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()

if __name__ == '__main__':

    print('test {}'.format(C.test_kitti_ckp_name))

    ckp_sta, ckp_end = 0, 80
    records = np.zeros([ckp_end-ckp_sta, 8])



    for i in range(ckp_sta, ckp_end):
        generate_prediction_disp(i, 'stereo', 'kitti')
        pred_disps = np.load("./test_disps.npy")
        print("epoch {}:".format(i))
        records[i] = compute_kitti_disp_errors(pred_disps)

        #compute_eigen_disp_errors(pred_disps)

    errors_name = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'D1', 'a1', 'a2', 'a3']
    data_df = pd.DataFrame(records, columns=errors_name)
    writer = pd.ExcelWriter(C.test_kitti_ckp_name + '_disp.xlsx')
    data_df.to_excel(writer, 'disp')
    writer.save()

