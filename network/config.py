
from torch.utils.data import DataLoader

class Config(object):

    #gpu_idx = '7'

    kitti_raw_dataset_path = '/home2/Dave/downloads/kitti_raw/'
    kitti_sf_2012_dataset_path = '/home2/Dave/downloads/kitti_scene_flow_2012/'
    kitti_sf_2015_dataset_path = '/home2/Dave/downloads/kitti_scene_flow_2015/'

    # eigen: eigen_cycle_8000, kitti: kitti_train_files_png_4frames
    filename_path = '/home2/Dave/pytorch/athena/utils/filenames/kitti_train_files_png_4frames.txt'
    # checkpoint load
    load_ckp_name = 'kitti_e80_stereo-flow_f_ir_gs'
    # checkpoint save path & name
    checkpoint_path = '/home2/Dave/pytorch/athena/ckp/'
    #checkpoint_name = 'kitti_e80_b2_s-f-c_f-b_scale-disp_ir_gs-20' #
    checkpoint_name = 'test'
    # tensorboard logger name
    logger_suffix= '_' + checkpoint_name

    num_thread = 6
    input_height = 192
    input_width = 640
    batch_size = 2
    num_epoch = 80
    num_scales = 4
    scales = [0, 1, 2, 3]

    is_pretrained = True
    num_layers = 18

    #loss
    learning_rate = 1e-4
    alpha_image_loss = 0.85
    disp_gradient_loss_weight = 0.1
    lamda_sm = 10
    lamda_lr = 0.5

    # test checkpoint, test filenames
    # training:
    #   191003: kitti_e80_b2_s-f-c_f-b_ori-disp_ir_gs
    #   191004: kitti_e80_b2_s-f-c_f-b_scale-disp_ir_gs-20

    test_kitti_ckp_name = 'kitti_e80_b2_s-f-c_f-b_scale-disp_ir_gs-20'
    test_eigen_ckp_name = 'kitti_e80_b2_s-f-c_f-b_ori-disp_ir_gs'
    test_stereo_eigen_filename_path = '/home2/Dave/pytorch/athena/utils/filenames/eigen_test_files.txt'
    test_stereo_kitti_filename_path = '/home2/Dave/pytorch/athena/utils/filenames/kitti_stereo_2015_test_files.txt'

    test_pwc_kitti_ckp_name = 'kitti_pwc_e80_b2_all_f-b_ir_gs/'



