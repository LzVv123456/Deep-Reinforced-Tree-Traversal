import os
import nrrd
import ast
import numpy as np
from collections import namedtuple
from torch.utils.data import DataLoader, Dataset


class GetDateset(Dataset):
    
    def __init__(self, directory, cfgs):
        self.directory = directory
        self.cfgs = cfgs
        self.resample_dist = self.cfgs.delay_update*self.cfgs.step_size

    def __len__(self):
        length = len(self.directory)
        return length

    def __getitem__(self, idx):

        single_dataset = read_single_dataset(self.directory[idx])
        image = single_dataset['image']
        tree = single_dataset['tree']

        # resample centerline and rerange image
        image = rerange(image)
        tree = resample_tree(tree, resample_dist=self.resample_dist)

        single_dataset['image'] = image
        single_dataset['tree'] = tree

        return single_dataset


def read_single_dataset(directory):

    single_dataset = {}

    dataset_name = os.path.basename(os.path.normpath(directory))
    dir_name = directory

    print('loading......', directory)
    single_dataset['name'] = directory

    # get image
    img, head = nrrd.read(dir_name + '/image_resample.nrrd')

    # get vessel tree
    with open(dir_name + '/tree_s.txt') as f:
        tree = f.read()
        tree = ast.literal_eval(tree)
        f.close()

    single_dataset['image'] = img
    single_dataset['tree'] = tree

    with open(dir_name + '/initial-point.txt') as f:
        start_points = {}
        sp = f.read().splitlines()
        start_points['l'] = ast.literal_eval(sp[0][2:])
        start_points['r'] = ast.literal_eval(sp[1][2:])
        print('start points: {}'.format(start_points))
        single_dataset['start_pts'] = start_points
        f.close()
    
    return single_dataset


def rerange(image):
    # map CT value to (-100, 700), equal to level 300, window 800
    new_min = -100
    new_max = 700
    new_image = np.clip(image, new_min, new_max)
    new_image = (new_image + 100)/800
    return new_image


def resample_tree(tree, resample_dist=2):
    new_tree = {}
    
    for key, vessel_centerline in tree.items():
        centerline_resample = []
        centerline_resample.append(list(vessel_centerline[0]))
        current_coordinate = vessel_centerline[0]
        index = 0
        
        while index <= len(vessel_centerline)-1:
            if index == len(vessel_centerline)-1:
                centerline_resample.append(list(vessel_centerline[-1]))
                break
            index_point = vessel_centerline[index]
            dist = np.linalg.norm(np.array(index_point) - np.array(current_coordinate))

            if dist >= resample_dist:
                scale = resample_dist/dist
                x_dist = index_point[0] - current_coordinate[0]
                y_dist = index_point[1] - current_coordinate[1]
                z_dist = index_point[2] - current_coordinate[2]

                next_coordinate = [current_coordinate[0] + scale*x_dist, 
                                   current_coordinate[1] + scale*y_dist, 
                                   current_coordinate[2] + scale*z_dist]
                centerline_resample.append(next_coordinate)
                current_coordinate = next_coordinate

                if dist < 2*resample_dist:
                    index += 1
            else:
                index += 1
        # append new centerline to tree dict
        new_tree[key] = centerline_resample

    return new_tree


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class PrioritizedReplay(object):

    def __init__(self, capacity, prob_alpha=0.6, beta_start=0.4, beta_frames=100000, frame=1):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = frame

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state):
        # assert state.ndim == next_state.ndim
        assert state.size() == next_state.size()
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs  = np.power(prios, self.prob_alpha)
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch = Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)