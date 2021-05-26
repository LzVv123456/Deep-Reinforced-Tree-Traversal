import ast
import torch
import numpy as np
from operator import itemgetter, attrgetter
from numpy import pi, cos, sin, arccos, arange
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

def get_ref_orientation(output_num, step_size):
    num_pts = output_num
    indices = arange(0, num_pts, dtype=float) + 0.5

    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices
    x, y, z = cos(theta)*sin(phi)*step_size, sin(theta)*sin(phi)*step_size, cos(phi)*step_size;

    ref_vector_list = []
    for i in range(len(x)):
        ref_vector_list.append(np.asarray([x[i], y[i], z[i]]))

    return np.asarray(ref_vector_list)


def get_match_rate(tree_list, trace_list, match_dist=8):
    match_rate = 0
    for pt in tree_list:
        dist = np.linalg.norm(np.array(trace_list) - np.array(pt), axis=1)
        min_dist = np.amin(dist)
        if min_dist < match_dist:
            match_rate += 1
    
    match_rate/= len(tree_list)
    return match_rate

def prepare_stat_dict(stat_dict, name):
    if name in stat_dict:
        pass
    else:
        stat_dict[name] = {} 
        stat_dict[name]['train time'] = 0
        stat_dict[name]['LCA progress'] = []
        stat_dict[name]['RCA progress'] = []
        stat_dict[name]['ALL progress'] = []
        stat_dict[name]['LCA average finish rate'] = 0
        stat_dict[name]['RCA average finish rate'] = 0
        stat_dict[name]['ALL average finish rate'] = 0

    stat_dict[name]['train time'] +=1
    return stat_dict


def prepare_training_area(start_pts):
    training_list = []

    for key, value in start_pts.items():
        for pt in value:
            training_list.append((pt, key))

    return training_list


def get_region_tree(start_idx, tree):
    edges = []
    for key in tree.keys():
        edges.append(key)

    whole_tree = []
    tree_segment = []
    jump_list = []
    jump_list.append(start_idx)

    while jump_list:
        current_idx = jump_list.pop()
        for key in tree.keys():
            if key in edges:
                if current_idx in ast.literal_eval(key):

                    if ast.literal_eval(key)[0] == current_idx:
                        next_idx = ast.literal_eval(key)[1]
                        tem_edge = tree[key]
                    else:
                        next_idx = ast.literal_eval(key)[0]
                        tem_edge = tree[key][::-1]
                    
                    tree_segment.append(tem_edge)
                    jump_list.append(next_idx)

                    for pt in tem_edge:
                        whole_tree.append(pt)

                    edges.remove(key)
                    
    return whole_tree, tree_segment


def tensor_to_numpy(sample):
    env = sample['image']
    tree = sample['tree']
    start_pts = sample['start_pts']
    name = sample['name']

    # from tensor to numpy
    env = env.to('cpu').numpy()
    env = np.squeeze(env, axis=0)

    new_tree = {}
    for key, value in tree.items():
        new_value = []
        for pt in value:
            new_value.append([pt[0].cpu().numpy()[0], pt[1].cpu().numpy()[0], pt[2].cpu().numpy()[0]])
        new_tree[key] = new_value

    l_list = []
    r_list = []
    for pt in start_pts['l']:
        l_list.append(pt.cpu().numpy()[0])
    for pt in start_pts['r']:
        r_list.append(pt.cpu().numpy()[0])
    start_pts['l'] = l_list
    start_pts['r'] = r_list

    name = name[0]
    
    return env, new_tree, start_pts, name


def crop_3d_volume(device, target_image, coordinate, patch_size):

    location = [int(x) for x in coordinate]

    if patch_size % 2 == 0:
        crop_size = int(patch_size / 2)

        x_top = location[0] + crop_size
        x_bottom = location[0] - crop_size
        y_top = location[1] + crop_size
        y_bottom = location[1] - crop_size
        z_top = location[2] + crop_size
        z_bottom = location[2] - crop_size

    elif patch_size % 2 == 1:
        crop_size = int((patch_size-1)/ 2)

        x_top = location[0] + crop_size + 1
        x_bottom = location[0] - crop_size
        y_top = location[1] + crop_size + 1
        y_bottom = location[1] - crop_size
        z_top = location[2] + crop_size + 1 
        z_bottom = location[2] - crop_size

    x_max, y_max, z_max = np.shape(target_image)

    if x_top >= x_max:
        x_top_1 = x_max
    else:
        x_top_1 = x_top

    if x_bottom <= 0:
        x_bottom_1 = 0
    else:
        x_bottom_1 = x_bottom

    if y_top >= y_max:
        y_top_1 = y_max
    else:
        y_top_1 = y_top

    if y_bottom <= 0:
        y_bottom_1 = 0
    else:
        y_bottom_1 = y_bottom

    if z_top >= z_max:
        z_top_1 = z_max
    else:
        z_top_1 = z_top

    if z_bottom <= 0:
        z_bottom_1 = 0
    else:
        z_bottom_1 = z_bottom


    cropped_volume = target_image[x_bottom_1:x_top_1, y_bottom_1:y_top_1, z_bottom_1:z_top_1]

    out_of_range = [x for x in np.shape(cropped_volume) if x != patch_size]

    if out_of_range:

        if x_top > x_max:
            x_top_pad = x_top - x_max
        else:
            x_top_pad = 0

        if x_bottom < 0:
            x_bot_pad = crop_size - location[0]
        else:
            x_bot_pad = 0

        if y_top > y_max:
            y_top_pad = y_top - y_max
        else:
            y_top_pad = 0

        if y_bottom < 0:
            y_bot_pad = crop_size - location[1]
        else:
            y_bot_pad = 0

        if z_top > z_max:
            z_top_pad = z_top - z_max
        else:
            z_top_pad = 0

        if location[2] - crop_size < 0:
            z_bot_pad = crop_size - location[2]
        else:
            z_bot_pad = 0

        cropped_volume = np.pad(cropped_volume, pad_width = ((x_bot_pad, x_top_pad), (y_bot_pad, y_top_pad), (z_bot_pad, z_top_pad)),
        mode='constant', constant_values=0)

    cropped_volume = np.expand_dims(np.expand_dims(cropped_volume, axis=0), axis=0)

    return torch.from_numpy(cropped_volume).to(device, dtype=torch.float)