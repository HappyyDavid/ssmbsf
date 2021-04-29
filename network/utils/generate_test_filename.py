
kitti_raw_dataset_path = '/home2/Dave/downloads/kitti_raw/'
kitti_sf_2012_dataset_path = '/home2/Dave/downloads/kitti_scene_flow_2012/'
kitti_sf_2015_dataset_path = '/home2/Dave/downloads/kitti_scene_flow_2015/'

from pathlib import Path

kitti_sf_2015_dataset = Path(kitti_sf_2015_dataset_path)

if kitti_sf_2015_dataset.exists():
    kitti_sf_2015_dataset_training_image_2 = kitti_sf_2015_dataset/'training'/'image_2'
    img2_prefix = 'training/image_2/'
    img3_prefix = 'training/image_3/'
    # find filename in image_2 folder
    filenames_path = sorted(kitti_sf_2015_dataset_training_image_2.iterdir())

    with open('athena_kitti_stereo_2015_test_file.txt', 'w+') as f:
        for idx in range(200):
            filename1 = filenames_path[2*idx].parts[-1]
            filename2 = filenames_path[2*idx+1].parts[-1]

            l_u = img2_prefix + filename1
            l_d = img2_prefix + filename2
            r_u = img3_prefix + filename1
            r_d = img3_prefix + filename2
            # not add '\n' in last raw
            if idx < 199:
                f.write(l_u + ' ' + r_u + ' ' + l_d + ' ' + r_d + '\n')
            else:
                f.write(l_u + ' ' + r_u + ' ' + l_d + ' ' + r_d)






