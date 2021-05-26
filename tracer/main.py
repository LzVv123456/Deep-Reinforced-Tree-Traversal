import os
import glob
import yaml
import torch
import argparse
from addict import Dict
from dataset import *
from init import *
from utilities import *
from train import *


def parse_args():
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--config', type=str, default='./tracer/train_config.yaml', 
                        help='path to config file')
    return parser.parse_args()


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    cfgs = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(cfgs)

    # get train, val dataset
    train_data_path = sorted(glob.glob(cfgs.train_path + '/**/'))
    train_set = GetDateset(train_data_path, cfgs)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

    # initialize everything
    policy_net, target_net, optimizer, scheduler, memory, steps_done, start_epoch = \
    inilization(cfgs, args)
    state_dict = {}
    all_trace_length = 0
    all_tree_length = 0

    # train epoch
    for epoch in range(start_epoch, cfgs.epoch):
        print('epoch: {}'.format(epoch))

        for _, sample in enumerate(train_loader): 
            env, tree, start_pts, name = tensor_to_numpy(sample)

            # prepare training regions and data mode
            training_list = prepare_training_area(start_pts)

            # save some statistic values 
            state_dict = prepare_stat_dict(state_dict, name)
            all_trace_length = 0
            all_tree_length = 0

            for item in training_list:
                print('training information', item)
                start_num, region = item

                traing_agent = Training_Agent(args, cfgs, target_net, policy_net, 
                env, tree, start_num, steps_done, optimizer, scheduler, memory)

                target_net, policy_net, trace_trajectory, STEPS_DONE = traing_agent.train()

                region_tree, _ = get_region_tree(start_num, tree)

                match_rate = get_match_rate(region_tree, trace_trajectory, cfgs.match_dist)
                print('match rate', np.round(match_rate * 100, 2))

                if region == 'l':
                    state_dict[name]['LCA progress'].append(np.round(match_rate*100, 2))
                elif region == 'r':
                    state_dict[name]['RCA progress'].append(np.round(match_rate*100, 2))

                all_tree_length += len(region_tree)
                all_trace_length += len(region_tree) * match_rate

            all_finish_rate = np.round(all_trace_length / all_tree_length, 2) * 100
            state_dict[name]['ALL progress'].append(all_finish_rate)

            if len(state_dict[name]['LCA progress']) > 0:
                state_dict[name]['LCA average finish rate'] = sum(state_dict[name]['LCA progress'])/len(state_dict[name]['LCA progress'])
            if len(state_dict[name]['RCA progress']) > 0:
                state_dict[name]['RCA average finish rate'] = sum(state_dict[name]['RCA progress'])/len(state_dict[name]['RCA progress'])
            state_dict[name]['ALL average finish rate'] = sum(state_dict[name]['ALL progress'])/len(state_dict[name]['ALL progress'])

        # print stat dict
        for key in sorted(state_dict.keys()):
            print(key, state_dict[key])

        # Update the target network
        if epoch % cfgs.update_epoch == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # save model
        if (epoch+1)%cfgs.save_freq==0:
            if not os.path.exists(cfgs.save_path):
                os.makedirs(cfgs.save_path)

            torch.save({
                'model_state_dict': target_net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'frames': memory.frame,
                'steps': steps_done,
                'epochs': epoch  
                }, cfgs.save_path + '/agent_' + str(epoch+1) + '.pth')

if __name__ == '__main__':
    args = parse_args()
    main(args)