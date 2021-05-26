import torch
import math
import random
import copy
import numpy as np


def select_action(device, last_action, policy_net, state, steps_done, eps_start, eps_end, eps_decay, output_channel):

    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    sample = random.random()

    if sample > eps_threshold:
        with torch.no_grad():
            policy_net.eval()
            return policy_net(state).max(1)[1].view(1, 1)    
    else:
        return torch.tensor([[random.randrange(output_channel)]], device=device, dtype=torch.long)


def update_env(env, current_location, action_idx, ref_orientation, mark_width=3, step_size=1):
    assert len(current_location) == 3, 'Coordinate is not 3D!'
    assert 3%2 == 1, 'wrong width! Must be odd number!'

    copy_current_location = copy.deepcopy(current_location)
    action_idx = action_idx.item()
    next_location = list(np.asarray(current_location) + np.asarray(ref_orientation[action_idx]))
    current_location = [int(x) for x in current_location]

    # Draw tractory on the env to represent state change (visual perspective)
    # Here we draw a cube with volume = (3*3*3)
    size = int((mark_width-1)/2) 
    mark = 3.0
    # draw point on image
    env[current_location[0]-size:current_location[0]+size+1, current_location[1]-size:current_location[1]+size+1,
    current_location[2]-size: current_location[2]+size+1] = mark

    return env, next_location


def get_reward(device, gt_list, current_location, next_location, match_dist, delay_update, step_size, forsee=1, k=1, verbal=False):
    
    reward_list = []
    geo_list = []
    pt_1 = np.array(current_location)
    pt_2 = np.array(next_location)

    if verbal:
        print('current_location', current_location)
        print('next_location', next_location)

    for gt in gt_list:
        gt = np.array(gt)
        dist_1 = np.linalg.norm(gt - pt_1, axis=1)
        dist_2 = np.linalg.norm(gt - pt_2, axis=1)

        idx_1 = np.argmin(dist_1)
        idx_2 = np.argmin(dist_2)
        pp_dist_1 = np.amin(dist_1)
        pp_dist_2 = np.amin(dist_2)
        pp_dist_avg =  (pp_dist_1+pp_dist_2)/2

        if verbal:
            print('p2p dist', pp_dist_avg)

        # prepare forsight index list
        index_list = []
        if forsee <= 1:
            forsee = 1
        for i in range(forsee):
            tem = idx_1 + (i+1)*k

            # prevent forsight index go out of max range
            if tem >= len(gt)-1:
                tem = len(gt)-1

            index_list.append(tem)

        # get trend
        weighted_trend = 0.0

        if pp_dist_avg < match_dist:
            for i in range(forsee):
                trend = (np.linalg.norm(current_location - gt[index_list[i]]) - np.linalg.norm(next_location - gt[index_list[i]]))\
                /(step_size*delay_update*2)
                weighted_trend += (1/forsee) * trend
        else:
            weighted_trend = 0.0

        # get reward
        if weighted_trend > 0: 
            # add a soft limit
            soft_limit = 1/(1+np.exp(pp_dist_avg))
            reward = weighted_trend + soft_limit
        else:
            reward = 0.0

        # clip the reward for safety
        reward = np.clip(reward, -1, 1)
        if verbal:
            print('tem_reward', reward)

        # append reward to reward list
        reward_list.append(reward)
        geo_list.append((idx_1, idx_2, pp_dist_1, pp_dist_2))

    return reward_list, geo_list
