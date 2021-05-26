import yaml
import glob
import argparse
from addict import Dict
from torch.utils.data import DataLoader
from dataset import *
from model import *
from validation import *


def parse_args():
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--config', type=str, default='./tracer/infer_config.yaml', 
                        help='path to config file')
    return parser.parse_args()


def infer(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    cfgs = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(cfgs)

    # prepare dataset
    val_data_path = sorted(glob.glob(cfgs.data_path + '/**/'))
    val_dataset = GetDateset(val_data_path, cfgs)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # load agent model
    agent = DQN_dila(cfgs)
    checkpoint = torch.load(cfgs.agent_model)
    agent.load_state_dict(checkpoint['model_state_dict'])
    if isinstance(agent, torch.nn.DataParallel):
            agent = agent.module
    agent.eval().to(args.device)
    print(agent)

    # load discriminator model
    discriminator = Classify_Dila(cfgs)
    checkpoint = torch.load(cfgs.discriminator_model)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    if isinstance(discriminator, torch.nn.DataParallel):
            discriminator = discriminator.module
    discriminator.to(args.device).eval()
    print(discriminator)

    # inference
    with torch.no_grad():
        # create evaluate agent
        evaluate_agent = Evaluate_Agent(cfgs, val_loader, agent, discriminator, args.device)
        all_point_val_result, all_point_rate_result, all_time_list = evaluate_agent.evaluate()


if __name__ == '__main__':
    args = parse_args()
    infer(args)