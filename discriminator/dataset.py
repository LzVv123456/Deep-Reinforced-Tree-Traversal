import random
import glob
import nrrd
import ast
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import affine_transform, zoom, rotate
from utilities import *


class Regression_Dateset(Dataset):
    
    def __init__(self, directory, patch_size, device, mode, augmentation=False):
        self.directory = directory
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.mode = mode
        self.pad_size = 5
        self.device = device
        self.coordinate_list, self.env_dict = prepare_data(self.directory, self.mode, self.patch_size)

    def __len__(self):

        length = len(self.coordinate_list)

        return length

    def __getitem__(self, idx):

        radius = 10
        coordinate, gt_bif, gt_vessel, bifurcation_list, reference_list, env_count = self.coordinate_list[idx]
        img = self.env_dict[env_count]

        if self.mode == 'train':

            # random shift
            if random.random() > 0.3:
                mean = coordinate
                cov = [[radius, 0, 0], [0, radius, 0], [0, 0, radius]] 
                x, y, z = np.random.multivariate_normal(mean, cov, 1).T
                coordinate = [x[0], y[0] ,z[0]]
                gt_bif, _ = get_regress_bif_gt(bifurcation_list, coordinate, self.patch_size)
                gt_vessel, _ = get_regress_vessel_gt(reference_list, coordinate, self.patch_size)

            # augmentation
            if self.augmentation:
                patch = implement_augmentation(coordinate, img, self.patch_size, self.pad_size)
            else:
                patch = roi_crop_pad(img, coordinate, self.patch_size)
        
        elif self.mode == 'val':
            patch = roi_crop_pad(img, coordinate, self.patch_size)

        return (patch, gt_bif, gt_vessel)


def prepare_data(directory, mode, patch_size):

    center_list, inwindow_list, offwindow_list = [], [], []
    env_dict = {}
    env_count = 0

    for dir_name in directory:
        print('loading data....')

        # get image
        img, head = nrrd.read(dir_name + '/image_resample.nrrd')
        img = rerange(img)

        env_dict[str(env_count)] = img

        # get vessel tree
        with open(dir_name + '/tree_s.txt') as f:
            tree = f.read()
            tree = ast.literal_eval(tree)
            f.close()

        # get all training data and gt
        list_tuple = (center_list, inwindow_list, offwindow_list)
        center_list, inwindow_list, offwindow_list = prepare_patch_gt(list_tuple, str(env_count), tree, patch_size)
        env_count += 1

    # balance different categories in traning mode
    print('length of three lists:', len(center_list), len(inwindow_list), len(offwindow_list))
    coordinate_list = []

    if mode == 'train':
        for i in range(3):
            for pt in center_list:
                coordinate_list.append(pt)

        for pt in inwindow_list:
            coordinate_list.append(pt)
        
        for idx, pt in enumerate(offwindow_list):
            coordinate_list.append(pt)
        
    elif mode == 'val':
        for pt_list in [center_list, inwindow_list, offwindow_list]:
            for pt in pt_list:
                coordinate_list.append(pt)

    return coordinate_list, env_dict


def prepare_patch_gt(list_tuple, env_count, tree, patch_size):
    ref_coordinates_list = []
    bifurcation_list = []
    center_list, inwindow_list, offwindow_list = list_tuple

    for key, value in tree.items():
        for coordinate in value:
            ref_coordinates_list.append(coordinate)
        
    for coordinate in ref_coordinates_list:
        if ref_coordinates_list.count(coordinate) >= 3:
            bifurcation_list.append(coordinate)
  
    for coordinate in ref_coordinates_list:
    
        gt_bif, min_dist = get_regress_bif_gt(bifurcation_list, coordinate, patch_size)
        gt_vessel = np.exp(6) - 1

        if min_dist <= 2:
            center_list.append((coordinate, gt_bif, gt_vessel, bifurcation_list, ref_coordinates_list, env_count))
        elif 2 < min_dist <= patch_size/2:
            inwindow_list.append((coordinate, gt_bif, gt_vessel, bifurcation_list, ref_coordinates_list, env_count))
        else:
            offwindow_list.append((coordinate, gt_bif, gt_vessel, bifurcation_list, ref_coordinates_list, env_count))
                    
    return (center_list, inwindow_list, offwindow_list)


def get_regress_bif_gt(bifurcation_list, pt, patch_size, a=6):

    dist = np.linalg.norm(np.asarray(bifurcation_list)-np.asarray(pt), axis=1)
    min_dist = np.amin(dist)

    d_m = patch_size/2

    if min_dist < d_m:
        gt = np.exp(a*(1-min_dist/d_m)) - 1
    else:
        gt = 0

    return gt, min_dist


def get_regress_vessel_gt(reference_list, pt, patch_size, a=6):

    dist = np.linalg.norm(np.asarray(reference_list)-np.asarray(pt), axis=1)
    min_dist = np.amin(dist)

    d_m = patch_size/2

    if min_dist < d_m:
        gt = np.exp(a*(1-min_dist/d_m)) - 1
    else:
        gt = 0

    return gt, min_dist


def implement_augmentation(coordinate, img, origin_patch_size, pad_size):

    patch_size = origin_patch_size + 2 * pad_size

    patch = roi_crop_pad(img, coordinate, patch_size)

    # rotation
    if random.random() >= 0.5:
        degree_range = 180
        degree = [random.randrange(-degree_range, degree_range), random.randrange(-degree_range, degree_range), random.randrange(-degree_range, degree_range)]
        patch = rotation(patch, degree)

    # # zoom
    # if random.random() >= 0.5:
    #     patch = random_zoom_3d(patch, crop_size, 0.2)

    # intensity shift
    # if random.random() >= 0.5:
    #     random_intensity_shift(patch, 0.1, 0.1)

    min_index = pad_size
    max_index = pad_size + origin_patch_size
    patch = patch[min_index:max_index, min_index:max_index, min_index:max_index]

    return patch


def rotation(patch, degree):
    patch = rotate(patch, degree[0], (1,2), reshape=False, mode='nearest')
    patch = rotate(patch, degree[1], (0,2), reshape=False, mode='nearest')
    patch = rotate(patch, degree[2], (0,1), reshape=False, mode='nearest')
    return patch


def random_zoom_3d(volume, crop_size, max_scale_deltas):

    scale = random.uniform(1 - max_scale_deltas, 1 + max_scale_deltas)
    volume_zoom = zoom(volume, scale, order=0)

    new_center = np.round(np.array(np.shape(volume_zoom)) / 2)

    if scale > 1:
        center = int(new_center[0])
        volume_zoom = volume_zoom[center-crop_size:center+crop_size, center-crop_size:center+crop_size, center-crop_size:center+crop_size]

    elif scale < 1:
        center = int(new_center[0])
        pad_left = crop_size - center
        pad_right = crop_size - (np.shape(volume_zoom)[0] - center)
        volume_zoom = np.pad(volume_zoom, pad_width = ((pad_left,pad_right), (pad_left,pad_right), (pad_left,pad_right)), mode='edge')
    else:
        pass
        
    return volume_zoom


def random_intensity_shift(volume, max_offset, max_scale_delta):

    offset = random.uniform(-max_offset, max_offset)
    scale = random.uniform(1 - max_scale_delta, 1 + max_scale_delta)

    volume += offset
    volume *= scale

    return volume


def split_dataset(train_num, val_num, data_dir, seed=1):
    
    random.seed(seed)

    all_datasets = sorted(glob.glob(data_dir + '/**/**/'))

    random.shuffle(all_datasets)

    train_dirs = all_datasets[:train_num]
    val_dirs = all_datasets[train_num:train_num+val_num]
    test_dirs = all_datasets[train_num+val_num:]

    return train_dirs, val_dirs, test_dirs
