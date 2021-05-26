import ast
import torch
import numpy as np
from utilities import *
from agent_env import *
from model import *


class Training_Agent(object):

    def __init__(self, args, cfgs, target_net, policy_net, env, tree, start_num, 
    steps_done, optimizer, scheduler, memory):

        self.target_net = target_net
        self.policy_net = policy_net
        self.env = env
        self.tree = tree
        self.start_num = start_num
        self.steps_done = steps_done 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.memory = memory

        self.patch_size = cfgs.input_size[0]
        self.device = args.device
        self.delay_update = cfgs.delay_update
        self.dead_coe = cfgs.dead_coe
        self.output_channel = cfgs.output_channel
        self.eps_start = cfgs.eps_start 
        self.eps_end = cfgs.eps_end 
        self.eps_decay = cfgs.eps_decay
        self.step_size = cfgs.step_size
        self.forsee = cfgs.forsee
        self.k = cfgs.k
        self.dyn_r_zone = cfgs.dyn_r_zone
        self.batch_size = cfgs.batch_size
        self.gamma = cfgs.gamma
        self.stop_dist = cfgs.stop_dist
        self.match_dist = cfgs.match_dist

        self.edge_list = [ast.literal_eval(x) for x in self.tree.keys()]
        self.jump_list = []
        self.trace_trajectory = []
        self.ref_orientation = get_ref_orientation(self.output_channel, self.step_size)
        self.current_idx = None
        self.last_action = None
    

    def train(self):

        current_location = self.prepare_start_location()
        self.jump_list.append((self.start_num, current_location))

        while self.jump_list:
            print('new segment')

            self.current_idx, current_location = self.jump_list.pop()

            gt_list, s_e_list, archieve_reward_list = self.prepare_equal_gt(current_location)
            
            if not gt_list:
                break

            current_state = crop_3d_volume(self.device, self.env, current_location, self.patch_size)

            self.last_action = None
            max_step = max([len(x) for x in gt_list]) * self.dead_coe

            for step in range(max_step):

                self.steps_done += 1
                current_location = list(np.round(current_location, 2))
                
                # save trajectory and solve loop
                if current_location not in self.trace_trajectory:
                    self.trace_trajectory.append(current_location)

                # delay action
                next_location, action = self.delay_action(current_state, current_location)

                # get reward list
                reward_list, geo_list = get_reward(self.device, gt_list, 
                current_location, next_location, self.match_dist, self.delay_update, 
                self.step_size, forsee=self.forsee, k=self.k)

                # select true reward and decide gt path
                reward, geo_list, gt_list, s_e_list = self.get_reward_gt(step, 
                reward_list, geo_list, gt_list, s_e_list, archieve_reward_list)

                # observe new state
                next_state = crop_3d_volume(self.device, self.env, next_location, 
                patch_size=self.patch_size)

                # save memory
                self.save_memory(current_state, action, reward, next_state)

                # Perform one step of the optimization (on the target network)
                optimize_model(self.device, self.policy_net, self.target_net, 
                self.optimizer, self.scheduler, self.memory, self.batch_size, 
                self.gamma)

                # stop loop
                stop = self.stop_cateria(current_location, next_location, gt_list, 
                s_e_list, geo_list, step, max_step)

                if stop:
                    break

                # move to the next state    
                current_state = next_state
                current_location = next_location

        print('current tree training finished!')    
        return self.target_net, self.policy_net, self.trace_trajectory, self.steps_done


    def prepare_equal_gt(self, current_location):

        gt_list = []
        s_e_list = []
        archieve_reward_list = []

        for s_e_pair in self.edge_list:
            if self.current_idx in s_e_pair:

                self.jump_list.append((self.current_idx, current_location))

                gt = self.tree[str(s_e_pair)]

                if s_e_pair.index(self.current_idx)==1:
                    gt = gt[::-1]

                s_e_list.append(s_e_pair)
                gt_list.append(gt)
                archieve_reward_list.append([])

        return gt_list, s_e_list, archieve_reward_list


    def prepare_start_location(self):
        for key in self.tree.keys():
            key_tuple = ast.literal_eval(key)
            if self.start_num in key_tuple:
                if self.start_num == key_tuple[0]:
                    idx = 0
                else:
                    idx = -1

                start_location = self.tree[key][idx]
                break
        
        # random jitter at init location
        mean = start_location
        cov = [[4, 0, 0], [0, 4, 0], [0, 0, 4]] 
        x, y, z = np.random.multivariate_normal(mean, cov, 1).T
        start_location = [x[0], y[0] ,z[0]]

        return start_location


    def delay_action(self, current_state, current_location):
        tem_current_state = current_state
        tem_current_location = current_location
        action = []

        for i in range(self.delay_update):
            tem_action = select_action(self.device, self.last_action, self.policy_net, tem_current_state,
            self.steps_done, self.eps_start, self.eps_end, self.eps_decay, self.output_channel)

            self.last_action = tem_action

            self.env, tem_next_location = update_env(self.env, tem_current_location, 
            tem_action, self.ref_orientation,step_size=self.step_size)

            tem_next_state = crop_3d_volume(self.device, self.env, tem_next_location, patch_size=self.patch_size)

            tem_current_state = tem_next_state

            tem_current_location = tem_next_location

            action.append(tem_action.item())
        
        next_location = tem_next_location
        action = torch.tensor([action])

        return next_location, action

    
    def get_reward_gt(self, step, reward_list, geo_list, gt_list, s_e_list, archieve_reward_list):

        max_reward = max(reward_list)
        print('reward', max_reward)

        if step <= self.dyn_r_zone:
            for idx, reward in enumerate(reward_list):
                archieve_reward_list[idx].append(reward)

        if step == self.dyn_r_zone:
            tem = [np.mean(x) for x in archieve_reward_list]
            idx = tem.index(max(tem))
            gt_list = [gt_list[idx]]
            s_e_list = [s_e_list[idx]]
            geo_list = [geo_list[idx]]

            # delete current index from jump list
            for jump_pair in self.jump_list:
                if self.current_idx == jump_pair[0]:
                    self.jump_list.remove(jump_pair)
                    break
                else:
                    pass

            # delete selected edge from edge list
            s_e_pair = s_e_list[0]
            self.edge_list.remove(s_e_pair)

        max_reward = torch.tensor([max_reward], device=self.device)
            
        return max_reward, geo_list, gt_list, s_e_list 


    def save_memory(self, current_state, action, reward, next_state):
        self.memory.push(current_state.to('cpu'), action.to('cpu'), reward.to('cpu'), next_state.to('cpu'))


    def stop_cateria(self, current_location, next_location, gt_list, s_e_list, geo_list, step, max_step):
        stop = False

        for idx, gt in enumerate(gt_list):

            idx_1, idx_2, pp_dist_1, pp_dist_2 = geo_list[idx]
            pp_dist_avg =  (pp_dist_1+pp_dist_2)/2

            if (pp_dist_avg > self.stop_dist) and (step > self.dyn_r_zone):
                print('Out of safe range!')
                stop = True

            elif idx_1 >= len(gt)-1:
                print('Reach end of current segment!')
                stop = True

                if s_e_list[idx].index(self.current_idx) == 0:
                    jump_idx = s_e_list[idx][1]
                else:
                    jump_idx = s_e_list[idx][0]

                not_end = False
                for s_e_pair in self.edge_list:
                    if jump_idx in s_e_pair:
                        not_end = True
                
                if not_end:
                    self.jump_list.append((jump_idx, current_location))

            elif step >= max_step-1:
                stop = True

        return stop