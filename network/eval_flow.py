import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.io as io
from torch.utils.data import DataLoader

from athena_config import Config
C = Config()

from athena_dataloader import get_test_flow_data_path, EvaluateFlowDataset
from athena_resnet_encoder import ResnetEncoder
from athena_depth_decoder import DepthDecoder
from athena_loss import image_sequence, model_inputs
from bridge_evaluation_utils import *


filename_dir = "/home2/Dave/pytorch/athena/utils/filenames/"

filenames = {
    "noc_2012": "kitti_flow_val_files_194_2012.txt", "occ_2012": "kitti_flow_val_files_occ_194_2012.txt",
    "noc_2015": "kitti_flow_val_files_200.txt", "occ_2015": "kitti_flow_val_files_occ_200.txt",
}

encoder = ResnetEncoder(C.num_layers, False, 2)
decoder = DepthDecoder(encoder.num_ch_enc, C.scales, num_output_channels=2)
encoder.cuda()
decoder.cuda()

print("test {}".format(C.test_kitti_ckp_name))

ckp_sta, ckp_end = 0, 80
records = np.zeros([ckp_end-ckp_sta, 8])

for i in range(ckp_sta, ckp_end):
    print("epoch {}".format(i))
    # load model
    ckp_load_dir = os.path.join(C.checkpoint_path, C.test_kitti_ckp_name, "weights_{}".format(i))
    encoder_path = os.path.join(ckp_load_dir, "encoder.pth")
    decoder_path = os.path.join(ckp_load_dir, "decoder.pth")


    encoder_dict = torch.load(encoder_path)
    h = encoder_dict['height']
    w = encoder_dict['width']
    del encoder_dict['height']
    del encoder_dict['width']

    encoder.load_state_dict({k: v for k, v in encoder_dict.items()})
    decoder.load_state_dict(torch.load(decoder_path))

    encoder.eval()
    decoder.eval()

    result = []
    # loop kitti flow test
    for k, v in filenames.items():

        dataset_path = C.kitti_sf_2012_dataset_path if k[-4:] == '2012' else C.kitti_sf_2015_dataset_path
        data_path, flow_path = get_test_flow_data_path(filename_dir + filenames[k], dataset_path)

        dataset = EvaluateFlowDataset(data_path, flow_path, {'h': C.input_height, 'w': C.input_width})
        # flow files have different size, thus we use batch = 1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

        total_error = 0
        fl_error = 0
        num_test = 0

        # tmp_variable, delete latter
        # t_error_, t_fl_ = 0, 0
        # t__error, t__fl = 0, 0

        with torch.no_grad():
            for idx, (data, flow_data) in enumerate(dataloader):

                for key, item in data.items():
                    data[key] = item.cuda().expand([C.batch_size, -1, -1, -1]) # like batch = 2


                former, latter = image_sequence(data, mode='flow')
                forward_in, backward_in = model_inputs(former, latter)
                forward_features = encoder(forward_in)
                forward_out = decoder(forward_features)



                pred_flo = forward_out['disp', 0][0].unsqueeze(0)
                # test optim
                pred_flo_ = nn.UpsamplingBilinear2d(size=(flow_data['img_h'], flow_data['img_w']))(pred_flo)

                #np.save('{}_flo.npy'.format(idx), pred_flo_.detach().cpu().numpy())

                pred_flo_[0, 0] = pred_flo_[0, 0] * int(flow_data['img_w']) / 2
                pred_flo_[0, 1] = pred_flo_[0, 1] * int(flow_data['img_h']) / 2

                pred_flo = nn.UpsamplingBilinear2d(size=(flow_data['h'], flow_data['w']))(pred_flo)
                pred_flo[0, 0] = pred_flo[0, 0] * int(flow_data['w']) / 2
                pred_flo[0, 1] = pred_flo[0, 1] * int(flow_data['h']) / 2

                # TODO i think here is test code, the mask should be flow_data['mask']
                # mask = np.ceil(np.clip(np.abs(flow_data['flo'][0, 0]), 0, 1))
                # error, fl = evaluate_flow(pred_flo[0].detach().cpu().numpy(), flow_data['flo'][0].numpy(), mask.numpy())

                # ***************************- TEST -*************************** #
                # output .mat when test kitti_2015

                # if (k == 'noc_2015'):
                #     save_file_ = str(idx).zfill(6) + '_12l.mat'
                #     print(save_file_)
                #     save_npy = pred_flo_[0].permute(1, 2, 0).detach().cpu().numpy()
                #     io.savemat(save_file_, {"flow_12l": save_npy})

                # test optimized result
                # if (k == 'occ_2015') & (idx < 200):
                #     print(idx)
                #     flow_op = io.loadmat(str(idx).zfill(6)+'_10_flo_op.mat')['flow_12l_op']
                #     error_, fl_ = evaluate_flow(flow_op.transpose(2, 0, 1), flow_data['flo'][0].numpy(), flow_data['mask'].numpy())
                #     print(error_, fl_)
                #     t_error_= t_error_ + error_
                #     t_fl_ = t_fl_ + fl_
                #     _error, _fl = evaluate_flow(pred_flo[0].detach().cpu().numpy(), flow_data['flo'][0].numpy(),
                #                               flow_data['mask'].numpy())
                #     t__error = t__error + _error
                #     t__fl = t__fl + _fl
                #
                #     print(_error, _fl)
                #     #print('haha')
                #
                # if idx == 199:
                #     print(t_error_/200, t_fl_/200)
                #     print(t__error/200, t__fl/200)

                # ***************************- TEST -*************************** #



                error, fl = evaluate_flow(pred_flo[0].detach().cpu().numpy(), flow_data['flo'][0].numpy(), flow_data['mask'].numpy())

                total_error += error
                fl_error += fl
                num_test += 1

                #print(idx)

        total_error /= num_test
        fl_error /= num_test
        # record results
        result.append(total_error)
        result.append(fl_error)

        #print("The average EPE is : ", total_error)
        #print("The average Fl is : ", fl_error)
    print("{:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}".format(
            '12_noc_epe', '12_noc_fl', '12_occ_epe', '12_occ_fl', '15_noc_epe', '15_noc_fl', '15_occ_epe', '15_occ_fl'))
    print("{:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}, {:12.4f}".format(
        result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]))

    records[i] = result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7] # save result  to print

errors_name = ['12_noc_epe', '12_noc_fl', '12_occ_epe', '12_occ_fl', '15_noc_epe', '15_noc_fl', '15_occ_epe', '15_occ_fl']
data_df = pd.DataFrame(records, columns=errors_name)
writer = pd.ExcelWriter(C.test_kitti_ckp_name + '_flow.xlsx')
data_df.to_excel(writer, 'flow')
writer.save()