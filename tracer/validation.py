import torch
import copy
import time
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max
from utilities import *
from agent_env import *


class Evaluate_Agent():

    def __init__(self, cfgs, val_loader, target_net, classifier, device):

        self.val_loader = val_loader
        self.target_net = target_net
        self.classifier = classifier
        self.device = device

        self.region_mode = cfgs.region_mode
        self.start_idx = cfgs.start_idx
        self.input_size = cfgs.input_size
        self.delay_update = cfgs.delay_update
        self.output_channel = cfgs.output_channel
        self.step_size = cfgs.step_size
        self.match_dist = cfgs.match_dist
        self.plot = cfgs.plot
        self.scatter = cfgs.scatter
        self.dynamic = cfgs.dynamic_plot
        self.precise_metric = cfgs.precise_metric
        self.plot_mode = cfgs.plot_mode

        self.dataset_name = None
        self.bifurcation_draw = []
        self.momentum_len = 1
        self.q_history = collections.deque(maxlen = self.momentum_len)
 
        self.step_mm = self.delay_update*self.step_size*0.5 # distance for each step in mm
        self.regress_bif_value = collections.deque(maxlen = int(10/self.step_mm))
        self.regress_bif_coord = collections.deque(maxlen = int(10/self.step_mm))
        self.stop_count = int(6/self.step_mm)  # average entropy with in 6mm 
        self.stop_cache = collections.deque(maxlen = self.stop_count)
        self.stop_threshold = cfgs.stop_threshold  # threshold for stop
        self.bif_threshold = cfgs.bif_threshold  # threshold for bifurcation detection
        self.bif_maximum = 200  # maixmum sampled bifurcation num
        self.bif_count = 0  # current bifurcation num
        self.max_step = int(300/self.step_mm)  # maximum step for single branch
        self.ref_orientation = get_ref_orientation(self.output_channel, self.step_size)
        self.stop_dist = 6


    def evaluate(self):

        print('Start Validation')
        all_point_val_result = []
        all_point_rate_result = []
        all_time_list = []

        for _, sample in enumerate(self.val_loader):
            
            tpr, tpm, fn, fp, ai_gt, ai_infer, tpr_of, tpm_of, fn_of, fp_of = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            instance_parameter_tuple = (tpr, tpm, fn, fp, ai_gt, ai_infer, tpr_of, tpm_of, fn_of, fp_of)

            env, tree, start_pts, name = tensor_to_numpy(sample)
            print('dataset name:', name)
            ref_env = copy.deepcopy(env)

            # prepare all regions need to be validated
            val_list = prepare_training_area(start_pts)

            # instance level plot
            trajectory_draw = []
            start_location_draw = []
            self.bifurcation_draw = []

            instance_time = 0
            for item in val_list:

                print('validation information', item)

                s_time = time.time() # get start time

                start_num, region = item

                tree_gt, tree_gt_segment = get_region_tree(start_num, tree)

                end_list, bifurcation_list, jump_reference, start_location = self.prepare_bifurcation_end(tree_gt_segment, tree, start_num)

                trajectory = self.trace_vessel(env, ref_env, end_list, tree_gt, bifurcation_list, jump_reference, start_location)

                e_time = time.time() # get end time
                print('time consuming:', e_time - s_time) # time used for tracing current tree
                instance_time += (e_time - s_time)

                # prepare plot
                trajectory_draw.append(trajectory)
                start_location_draw.append(start_location)
                
                instance_parameter_tuple = self.calculate_loss(instance_parameter_tuple, tree_gt, tree_gt_segment, trajectory)

            point_result_tuple, rate_result_tuple = self.calculate_metric(instance_parameter_tuple)
            all_point_val_result.append(list(point_result_tuple)) # this collect detail result of all tree structure 
            all_point_rate_result.append(list(rate_result_tuple))
            all_time_list.append(instance_time)

            if self.plot:
                self.plot_trajectory(tree, trajectory_draw, start_location_draw)

        self.overal_result(all_point_val_result, all_point_rate_result, all_time_list)

        return all_point_val_result, all_point_rate_result, all_time_list


    def trace_vessel(self, env, ref_env, end_list, tree_gt, bifurcation_list, jump_reference, start_location):
        
        trace_trajectory = []
        jump_list = []
        self.bif_count = 0
        
        out_of_range = False
        first_branch = True
        jump_list.append(start_location)

        while jump_list:
            # clear cache
            self.q_history.clear()
            self.stop_cache.clear()
            self.regress_bif_coord.clear()
            self.regress_bif_value.clear()

            segment_trajectory = []
            current_location = jump_list.pop(0)

            self.bif_count += 1
            if self.bif_count >= self.bif_maximum:
                break

            current_state = crop_3d_volume(self.device, env, current_location, int(self.input_size[0]))

            last_action = None
            for step in range(self.max_step):
                current_location = list(np.round(current_location, 2))
                if (current_location[0] <= 0) or (current_location[0] >= np.shape(env)[0]):
                    out_of_range = True
                    break
                if (current_location[1] <= 0) or (current_location[1] >= np.shape(env)[1]):
                    out_of_range = True
                    break
                if (current_location[2] <= 0) or (current_location[2] >= np.shape(env)[2]):
                    out_of_range = True
                    break
                
                # save trajectory and solve loop
                if current_location not in segment_trajectory:
                    segment_trajectory.append(current_location)
                
                # propose next location
                next_location, last_action = self.delay_action(env, current_state, current_location, last_action)

                # Observe new state
                next_state = crop_3d_volume(self.device, env, next_location, patch_size=int(self.input_size[0]))

                # prepare jump
                ref_state = crop_3d_volume(self.device, ref_env, current_location, patch_size=int(self.input_size[0]))
                jump_list = self.auto_joint_discrimitor(jump_list, current_location, ref_state)
 
                # prepare stop
                if first_branch:
                    if step >= 50: # safe zone for first vessel
                        stop = self.auto_stop_function(current_location, end_list, step)
                        if stop:
                            break
                else:
                    if step >= 5: # safe zone for a new branch
                        stop = self.auto_stop_function(current_location, end_list, step)
                        if stop:
                            break

                # move to the next state    
                current_state = next_state
                current_location = next_location

            first_branch = False

            trace_trajectory, jump_list = self.prepare_trace_trajectory(trace_trajectory, segment_trajectory, jump_list)
            
            # trace go out of env
            if out_of_range:
                break
        
        return trace_trajectory


    def prepare_trace_trajectory(self, trace_trajectory, segment_trajectory, jump_list):
        end_idx = -1
        new_jump_list = copy.deepcopy(jump_list)

        if len(trace_trajectory)>1:
            if len(segment_trajectory) >= int(5/self.step_mm):
                cover_count = 0
                for pt in segment_trajectory:
                    for segment in trace_trajectory:
                        dist = np.linalg.norm(np.asarray(segment)-np.asarray(pt), axis=1)
                        min_dist = np.amin(dist)

                        if min_dist <= 6:
                            cover_count+=1
                            break

                cover_rate = cover_count/len(segment_trajectory)

                if cover_rate < 0.5:
                    trace_trajectory.append(segment_trajectory[:end_idx])
                else:
                    # remove jump pt on removed segment
                    for pt in jump_list:
                        if pt in segment_trajectory:
                            new_jump_list.remove(pt)            
            else:
                # remove jump pt on removed segment
                for pt in jump_list:
                    if pt in segment_trajectory:
                        new_jump_list.remove(pt)
        else:
            trace_trajectory.append(segment_trajectory[:end_idx])

        return trace_trajectory, new_jump_list


    def calculate_loss(self, parameter_tuple, tree_list, tree_segment, trace_segment):
            
        # get trace list
        trace_list = []
        for segment in trace_segment:
            for pt in segment:
                if pt not in trace_list:
                    trace_list.append(pt)

        tpr, tpm, fn, fp, ai_gt, ai_infer, tpr_of, tpm_of, fn_of, fp_of = parameter_tuple

        for segment in trace_segment:
            continious = True
            for pt in segment:
                dist = np.linalg.norm(np.asarray(tree_list) - np.asarray(pt), axis=1)

                if self.precise_metric:
                    min_idx = np.argmin(dist)
                    min_idx_0 = min_idx - 1
                    min_idx_1 = min_idx + 1

                    if min_idx_0 < 0:
                        min_idx_0 = 0
                    if min_idx_1 > len(tree_list)-1:
                        min_idx_1 = len(tree_list)-1
                    
                    tem_list = [tree_list[min_idx_0], tree_list[min_idx], tree_list[min_idx_1]]
                    ensampled_list = self.ensample_centerline(tem_list)
                    min_dist = np.amin(np.linalg.norm(np.asarray(ensampled_list) - np.asarray(pt), axis=1))
                else:
                    min_dist = np.amin(dist)

                if min_dist < self.match_dist:
                    tpm += 1
                    ai_infer += min_dist
                    if continious:
                        tpm_of += 1
                    else:
                        fp_of += 1
                else:
                    fp += 1
                    fp_of += 1
                    continious = False

        for segment in tree_segment:
            continious = True

            for pt in segment:

                dist = np.linalg.norm(np.asarray(trace_list) - np.asarray(pt), axis=1)

                if self.precise_metric:

                    min_idx = np.argmin(dist)
                    min_idx_0 = min_idx - 1
                    min_idx_1 = min_idx + 1

                    if min_idx_0 < 0:
                        min_idx_0 = 0
                    if min_idx_1 > len(trace_list)-1:
                        min_idx_1 = len(trace_list)-1

                    tem_list = [trace_list[min_idx_0], trace_list[min_idx], trace_list[min_idx_1]]
                    ensampled_list = self.ensample_centerline(tem_list)
                    min_dist = np.amin(np.linalg.norm(np.asarray(ensampled_list) - np.asarray(pt), axis=1))
                else:
                    min_dist = np.amin(dist)

                if min_dist < self.match_dist:
                    tpr += 1
                    ai_gt += min_dist
                    if continious:
                        tpr_of += 1
                    else:
                        fn_of += 1
                else:
                    fn += 1
                    fn_of += 1
                    continious = False
        
        return (tpr, tpm, fn, fp, ai_gt, ai_infer, tpr_of, tpm_of, fn_of, fp_of)


    def calculate_metric(self, parameter_tuple):

        tpr, tpm, fn, fp, ai_gt, ai_infer, tpr_of, tpm_of, fn_of, fp_of = parameter_tuple

        ov = np.round((tpr + tpm)/(tpr + tpm + fn + fp + 1e-8), 4)
        ov_single = np.round(tpr/(tpr + fn + 1e-8), 4)
        of = np.round((tpr_of + tpm_of)/(tpr_of + tpm_of + fn_of + fp_of + 1e-8), 4)
        ac = np.round(((ai_gt + ai_infer)/(tpr + tpm + 1e-8)) * 0.5, 4)
        ac_single = np.round((ai_gt/(tpr+1e-8)) * 0.5, 4)

        score = ov - ac
        print('Point Based ---------------------------------------------')
        print('Overlap_both: ', ov*100, '%')
        print('Overlap_single: ', ov_single*100, '%')
        print('Overlap Until First Error: ', of*100, '%')
        print('Inner Accuracy: ', ac, 'mm')
        print('Inner Accuracy Single: ', ac_single, 'mm')

        lambda1 = 0.5

        ov_gt = tpr/(tpr + fn + 1e-8)
        ov_infer = tpm/(tpm + fp + 1e-8)
        ov_rate = np.round(lambda1 * ov_gt + (1-lambda1) * ov_infer, 4)
        ov_single_rate = np.round(ov_gt, 4)

        ac_gt = (ai_gt/(tpr + 1e-8)) * 0.5
        ac_infer = (ai_infer/(tpm + 1e-8)) * 0.5
        ac_rate = np.round(lambda1 * ac_gt + (1-lambda1) * ac_infer, 4)
        ac_single_rate = np.round(ac_gt, 4)

        print('Rate Based ---------------------------------------------')
        print('Overlap_both: ', ov_rate*100, '%')
        print('Overlap_single: ', ov_single_rate*100, '%')
        print('Inner Accuracy: ', ac_rate, 'mm')
        print('Inner Accuracy Single: ', ac_single_rate, 'mm')
        print('----------------------------------------------------')
        return (score, ov, ov_single, of, ac, ac_single), (ov_rate, ov_single_rate, ac_rate, ac_single_rate)


    def overal_result(self, all_point_val_result, all_point_rate_result, all_time_list):

        assert len(all_point_val_result) == len(all_point_rate_result) == len(all_time_list), 'wrong metric!'

        p_ov, p_ov_single, p_of, p_ac, p_ac_single = [], [], [], [], []
        r_ov, r_ov_single, r_ac, r_ac_single = [], [], [], []

        for idx in range(len(all_point_val_result)):
            p_result = all_point_val_result[idx]
            r_result = all_point_rate_result[idx]

            p_ov.append(p_result[1])
            p_ov_single.append(p_result[2])
            p_of.append(p_result[3])
            p_ac.append(p_result[4])
            p_ac_single.append(p_result[5])

            r_ov.append(r_result[0])
            r_ov_single.append(r_result[1])
            r_ac.append(r_result[2])
            r_ac_single.append(r_result[3])

        print('Final Result: Max, Min, Mean !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Point Based ---------------------------------------------')
        print('Overlap both: {}% {}% {}%'.format(np.amax(p_ov)*100, np.amin(p_ov)*100, np.mean(p_ov)*100))
        print('Overlap single: {}% {}% {}%'.format(np.amax(p_ov_single)*100, np.amin(p_ov_single)*100, np.mean(p_ov_single)*100))
        print('Overlap Until First Error: {}% {}% {}%'.format(np.amax(p_of)*100, np.amin(p_of)*100, np.mean(p_of)*100))
        print('Inner Accuracy: {}mm {}mm {}mm'.format(np.amax(p_ac), np.amin(p_ac), np.mean(p_ac)))
        print('Inner Accuracy Single: {}mm {}mm {}mm'.format(np.amax(p_ac_single), np.amin(p_ac_single), np.mean(p_ac_single)))

        print('Rate Based ---------------------------------------------')
        print('Overlap both: {}% {}% {}%'.format(np.amax(r_ov)*100, np.amin(r_ov)*100, np.mean(r_ov)*100))
        print('Overlap single: {}% {}% {}%'.format(np.amax(r_ov_single)*100, np.amin(r_ov_single)*100, np.mean(r_ov_single)*100))
        print('Inner Accuracy: {}mm {}mm {}mm'.format(np.amax(r_ac), np.amin(r_ac), np.mean(r_ac)))
        print('Inner Accuracy Single: {}mm {}mm {}mm'.format(np.amax(r_ac_single), np.amin(r_ac_single), np.mean(r_ac_single)))

        print('Time ----------------------------------------------------')
        print('Time: {}% {}% {}%'.format(np.amax(all_time_list), np.amin(all_time_list), np.mean(all_time_list)))
        print('----------------------------------------------------')  

        # with open(str(self.stop_threshold)+'-'+str(self.bif_threshold)+'-'+str(self.step_size)+'.txt', 'w') as f:
        #     f.write('Final Result: Max, Min, Mean !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
        #     f.write('Point Based ---------------------------------------------\n')
        #     f.write('Overlap both: {}% {}% {}%\n'.format(np.amax(p_ov)*100, np.amin(p_ov)*100, np.mean(p_ov)*100))
        #     f.write('Overlap single: {}% {}% {}%\n'.format(np.amax(p_ov_single)*100, np.amin(p_ov_single)*100, np.mean(p_ov_single)*100))
        #     f.write('Overlap Until First Error: {}% {}% {}%\n'.format(np.amax(p_of)*100, np.amin(p_of)*100, np.mean(p_of)*100))
        #     f.write('Inner Accuracy: {}mm {}mm {}mm\n'.format(np.amax(p_ac), np.amin(p_ac), np.mean(p_ac)))
        #     f.write('Inner Accuracy Single: {}mm {}mm {}mm\n'.format(np.amax(p_ac_single), np.amin(p_ac_single), np.mean(p_ac_single)))
        #     f.write('Rate Based ---------------------------------------------\n')
        #     f.write('Overlap both: {}% {}% {}%\n'.format(np.amax(r_ov)*100, np.amin(r_ov)*100, np.mean(r_ov)*100))
        #     f.write('Overlap single: {}% {}% {}%\n'.format(np.amax(r_ov_single)*100, np.amin(r_ov_single)*100, np.mean(r_ov_single)*100))
        #     f.write('Inner Accuracy: {}mm {}mm {}mm\n'.format(np.amax(r_ac), np.amin(r_ac), np.mean(r_ac)))
        #     f.write('Inner Accuracy Single: {}mm {}mm {}mm\n'.format(np.amax(r_ac_single), np.amin(r_ac_single), np.mean(r_ac_single)))
        #     f.write('Time ----------------------------------------------------\n')
        #     f.write('Time: {}% {}% {}%\n'.format(np.amax(all_time_list), np.amin(all_time_list), np.mean(all_time_list)))
        #     f.write('----------------------------------------------------\n')   
        return None
        

    def ensample_centerline(self, vessel_centerline, resample_dist=0.05):

        centerline_resample = []
        centerline_resample.append(list(vessel_centerline[0]))
        current_coordinate = vessel_centerline[0]
        index = 0
        
        while index <= len(vessel_centerline)-1:

            if index == len(vessel_centerline)-1:
                break

            index_point = vessel_centerline[index]
            next_index_point = vessel_centerline[index+1]

            dist = np.linalg.norm(np.array(index_point) - np.array(current_coordinate))
            dist_next = np.linalg.norm(np.array(next_index_point) - np.array(index_point))

            if dist < dist_next:
                scale = resample_dist/dist_next
                x_dist = next_index_point[0] - index_point[0]
                y_dist = next_index_point[1] - index_point[1]
                z_dist = next_index_point[2] - index_point[2]
                next_coordinate = [current_coordinate[0]+scale*x_dist, current_coordinate[1]+scale*y_dist, current_coordinate[2]+scale*z_dist]
                centerline_resample.append(next_coordinate)
                current_coordinate = next_coordinate
            else:
                index += 1
            
        return centerline_resample


    def auto_stop_function(self, current_location, end_list, step):
        stop = False
        if step >= self.max_step - 1:
            stop = True
        if not stop:
            # remove largest value and smallest value
            ref_stop_cache = copy.deepcopy(self.stop_cache)
            ref_stop_cache.remove(np.amax(ref_stop_cache))
            ref_stop_cache.remove(np.amin(ref_stop_cache))

            avg_vessel_confidence = np.mean(ref_stop_cache)
            if avg_vessel_confidence <= self.stop_threshold:
                # print('not a vessel')
                stop = True

        return stop


    def auto_joint_discrimitor(self, jump_list, current_location, ref_state):
        outputs = self.classifier(ref_state.float())
        outputs = outputs.to('cpu').detach().numpy()[0]
        proximity_value = outputs[0].item()
        self.stop_cache.append(outputs[1].item())

        self.regress_bif_value.append(proximity_value)
        self.regress_bif_coord.append(current_location)

        min_dist = int(np.floor(1/self.step_mm)) # min_dist is always 1mm in real word 
        indices = peak_local_max(np.asarray(self.regress_bif_value), min_distance=min_dist, 
        threshold_abs=self.bif_threshold)

        for idx in indices:
            bifurcation_coordinate = self.regress_bif_coord[idx[0]]

            if len(jump_list) > 0:
                if bifurcation_coordinate not in jump_list:
                    jump_list.append(list(bifurcation_coordinate))
                    self.bifurcation_draw.append(list(bifurcation_coordinate))
            else:
                jump_list.append(list(bifurcation_coordinate))
                self.bifurcation_draw.append(list(bifurcation_coordinate))

        return jump_list


    def prepare_bifurcation_end(self, tree_segment, tree, start_num):
        jump_reference = []       
        bifurcation_list = []
        end_list = []

        for idx, vessel_segment in enumerate(tree_segment):
            if idx == 0:
                jump_reference.append(vessel_segment[self.start_idx])
            else:
                jump_reference.append(vessel_segment[0])

            jump_reference.append(vessel_segment[-1])
            
        start_location = None
        # prepare start location
        for key in tree.keys():
            key_tuple = ast.literal_eval(key)
            if start_num in key_tuple:
                segment = tree[key]
                if key_tuple.index(start_num) == 0:
                    pass
                else:
                    segment = segment[::-1]

                start_location = segment[self.start_idx]

        # remove start location
        if start_location and (start_location in jump_reference):
            jump_reference.remove(start_location)

        # prepare bifurcations and end points
        for pt in jump_reference:
            if (jump_reference.count(pt) == 1) and (pt not in end_list):
                end_list.append(pt)
                jump_reference.remove(pt)
            if (jump_reference.count(pt) >= 3) and (pt not in bifurcation_list):
                bifurcation_list.append(pt)
        
        return end_list, bifurcation_list, jump_reference, start_location


    def simple_action(self, current_state, current_location, last_action_idx):
        q_values = self.target_net(current_state).to(device='cpu').detach().numpy()[0]
        q_values = list(q_values)

        if self.q_history:
            self.q_history.append(q_values)
        else:
            for i in range(self.momentum_len):
                self.q_history.append(q_values)

        final_q_value = np.zeros((len(q_values),))
        unit_weight = 1/sum(x+1 for x in range(len(self.q_history)))

        for idx, value in enumerate(self.q_history):
            final_q_value += (idx+1) * unit_weight * np.asarray(value)

        if last_action_idx:

            # select action with angle less than 60 degrees againist last action
            last_action = self.ref_orientation[last_action_idx]
                
            indexed = list(enumerate(final_q_value))
            sorted_indexed = sorted(indexed, key=itemgetter(1), reverse=True)
            tem_idx = 0

            for pair in sorted_indexed:
                action_idx = pair[0]
                cur_action = self.ref_orientation[action_idx]
                # calculate angle between 2 vectors less than 60 degree
                cos_theta = np.dot(last_action, cur_action)/(np.linalg.norm(last_action) * np.linalg.norm(cur_action))
                if cos_theta >= 1/2:
                    break
            return torch.tensor(action_idx).to(self.device)

        else:
            action_idx = np.argmax(final_q_value)
            return torch.tensor(action_idx).to(self.device)


    def delay_action(self, env, current_state, current_location, last_action):
        tem_current_state = current_state
        tem_current_location = current_location

        for i in range(self.delay_update):
            tem_action = self.simple_action(tem_current_state, tem_current_location, last_action)

            last_action = tem_action

            env, tem_next_location = update_env(env, tem_current_location, 
            tem_action, self.ref_orientation, step_size=self.step_size)

            tem_next_state = crop_3d_volume(self.device, env, tem_next_location, int(self.input_size[0]))

            tem_current_state = tem_next_state

            tem_current_location = tem_next_location

        next_location = tem_next_location
 
        return next_location, last_action


    def plot_trajectory(self, tree_draw, trajectory_draw, start_location_draw):
        plt.rcParams["figure.figsize"] = 12.8, 9.6
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(32, -40)

        # plot gt centerline
        for key, gt_segment in tree_draw.items():
            x_g, y_g, z_g = [], [], []
            for pt in gt_segment:
                x_g.append(pt[0])
                y_g.append(pt[1])
                z_g.append(pt[2])
            ax.plot3D(x_g, y_g, z_g, zdir='z', c = 'b', linestyle='dashed')

        if 'b' in self.plot_mode:
            # plot bifurcation
            x_b, y_b, z_b = [], [], []
            for pt in self.bifurcation_draw:
                x_b.append(pt[0])
                y_b.append(pt[1])
                z_b.append(pt[2])
            ax.scatter3D(x_b, y_b, z_b, zdir='z', c = '#76cd26', s=10, alpha=1)

        # plot start location
        if 's' in self.plot_mode:
            for start_location in start_location_draw:
                ax.scatter3D(start_location[0], start_location[1], start_location[2], zdir='z', c = 'blueviolet', s=30, alpha=1)

        if 'tra' in self.plot_mode:
            # plot traced trajectory
            for trajectory in trajectory_draw:

                if self.dynamic:
                    self.dynamic_plot(trajectory, fig, ax)
                else:
                    for trace_segment in trajectory:
                        x_p, y_p, z_p = [], [], []

                        for pt in trace_segment:
                            x_p.append(pt[0])
                            y_p.append(pt[1])
                            z_p.append(pt[2])

                        if self.scatter:
                            ax.scatter3D(x_p, y_p, z_p, zdir='z', c = 'r', s=0.7, alpha=0.7)
                        else:
                            ax.plot3D(x_p, y_p, z_p, zdir='z', c = 'r', alpha=0.7)
        plt.axis('off')
        plt.grid(False)
        plt.show(block=True)
        plt.savefig('test.png')

    
    def dynamic_plot(self, all_trajectories, fig, ax):
        data = []
        for trace_trajectory in all_trajectories:   
            x_dynamic, y_dynamic, z_dynamic = [], [], []       
            for pt in trace_trajectory:
                x_dynamic.append(pt[0])
                y_dynamic.append(pt[1])
                z_dynamic.append(pt[2])

            dat = []
            dat.append(x_dynamic)
            dat.append(y_dynamic)
            dat.append(z_dynamic)

            dat = np.asarray(dat)
            # Fifty lines of random 3-D lines
            data.append(dat)

        def update_lines(num, dataLines, lines):
            for line, data in zip(lines, dataLines):
                # NOTE: there is no .set_data() for 3 dim data...
                line.set_data(data[0:2, :num])
                line.set_3d_properties(data[2, :num])
            return lines

        for dat in data:
            lines = [ax.plot3D(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], c = 'r')[0]]
            line_ani = animation.FuncAnimation(fig, update_lines, fargs=([dat], lines), interval=0, blit=False)
            pause = (25 * len(list(dat)[0])) / 1000
            plt.pause(pause)
