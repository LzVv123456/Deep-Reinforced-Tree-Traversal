import torch.optim as optim
from model import *
from dataset import *
from torch.optim.lr_scheduler import LambdaLR


def inilization(cfgs, args):

    # init network
    policy_net = DQN_dila(cfgs)
    target_net = DQN_dila(cfgs)

    # set optm
    optimizer = optim.Adam(policy_net.parameters(), lr=cfgs.lr)

    # parallel if GPU > 1
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        policy_net = nn.DataParallel(policy_net)
        target_net = nn.DataParallel(target_net)
    else:
        print("Let's use CPU!")

    policy_net.to(args.device)
    target_net.eval().to(args.device)

    # set schd
    _lambda = lambda step: max(0.99 ** (step//(5e4)), 1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda = _lambda)

    steps_done = 0
    start_epoch = 0
    frames = 1

    # set memory replay
    memory = PrioritizedReplay(cfgs.memory_replay_size, prob_alpha=cfgs.priority_alpha, 
    beta_start=cfgs.priority_beta_start, beta_frames=cfgs.priority_beta_frames, 
    frame=frames)

    return policy_net, target_net, optimizer, scheduler, memory, steps_done, start_epoch
