import os
import numpy as np

from bridge_evaluation_utils import read_text_lines, read_file_data, generate_depth_map, get_focal_length_baseline

eigen_test_filename_path = '/home2/Dave/pytorch/athena/utils/filenames/eigen_test_files.txt'
dataset_path = '/home2/Dave/downloads/kitti_raw/'

eigen_test_gt_save_path = '/home2/Dave/pytorch/athena/test_gt/eigen/'
eigen_test_gt_depth_save_path = eigen_test_gt_save_path + 'depths/'

def generate_eigen_depth_gt():
    '''
    generate eigen split depth ground truth using the projected point cloud.
    also save imgage size, focal length and baseline
    :return:
    '''

    if not os.path.exists(eigen_test_gt_save_path):
        os.makedirs(eigen_test_gt_save_path)

    if not os.path.exists(eigen_test_gt_depth_save_path):
        os.makedirs(eigen_test_gt_depth_save_path)

    n_samples = 697
    eigen_test_files = read_text_lines(eigen_test_filename_path)
    gt_velo, gt_calib, im_sizes, im_files, cams = read_file_data(eigen_test_files, dataset_path)

    assert n_samples == len(gt_velo)

    focal_length = np.zeros((697, 1), dtype=np.float32)
    baseline = np.zeros((697, 1), dtype=np.float32)
    img_size = np.zeros((697, 2), dtype=np.float32)


    for idx in range(n_samples):

        camera_id = cams[idx]  # 2 is left, 3 is right
        depth = generate_depth_map(gt_calib[idx], gt_velo[idx], im_sizes[idx], camera_id, False, True)
        np.save(eigen_test_gt_depth_save_path + 'depth_{}.npy'.format(idx), depth)

        fl, bl = get_focal_length_baseline(gt_calib[idx], camera_id)
        focal_length[idx] = fl
        baseline[idx] = bl
        img_size[idx] = np.asarray(im_sizes[idx])


    np.save(eigen_test_gt_save_path + 'focal_length.npy', focal_length)
    np.save(eigen_test_gt_save_path + 'baseline.npy', baseline)
    np.save(eigen_test_gt_save_path + 'img_size.npy', img_size)

if __name__ == "__main__":

    generate_depth_map()

