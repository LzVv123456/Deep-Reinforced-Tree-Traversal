import argparse
import torch
import yaml
import torch.optim as optim
from addict import Dict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from dataset import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--config', type=str, default='./discriminator/train_config.yaml', 
                        help='path to config file')
    return parser.parse_args()


def initialization(device, input_size, output_channel):
    network = Dilated_Regress_Net(input_size, output_channel).to(device)
    
    # parallel if GPU > 1
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network = nn.DataParallel(network)
    else:
        print("Let's use CPU!")

    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-5)

    criterion = nn.MSELoss()
    lambda1 = lambda iteration: max(0.99 ** (iteration//1e4), 1e-5)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda1)
    return network, optimizer, scheduler, criterion


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    cfgs = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfgs = Dict(cfgs)

    # get train, val dataset
    train_dirs = sorted(glob.glob(cfgs.train_path + '/**/'))
    val_dirs = sorted(glob.glob(cfgs.val_path + '/**/'))
    train_set = Regression_Dateset(train_dirs, cfgs.input_size[0], args.device, 'train', cfgs.aug)
    val_set = Regression_Dateset(val_dirs, cfgs.input_size[0], args.device, 'val')
    train_loader = DataLoader(train_set, cfgs.batch_size, shuffle=True, num_workers=cfgs.num_workers)
    val_loader = DataLoader(val_set, cfgs.batch_size, shuffle=False, num_workers=cfgs.num_workers)

    # initialization
    network, optimizer, scheduler, criterion = initialization(args.device, cfgs.input_size[0], cfgs.output_channel)

    # train
    save_loss = 1e5
    train_stat_loss = []
    val_stat_loss = []

    for epoch in range(cfgs.epoch):
        print('epoch: {}'.format(epoch))
        train_loss = 0
        train_bif_loss = 0
        train_vessel_loss = 0
        val_loss = 0
        val_bif_loss = 0
        val_vessel_loss = 0

        network.train()

        for i, data in enumerate(train_loader, 0):
            print('>', sep=' ', end='', flush=True)

            patch, gt_bif, gt_vessel = data
            patch = patch.to(args.device).float()
            gt_bif = gt_bif.to(args.device).float()
            gt_vessel = gt_vessel.to(args.device).float()
            patch = patch.unsqueeze(1)

            optimizer.zero_grad()
            outputs = network(patch).squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)

            output_bif = outputs[:, 0]
            output_vessel = outputs[:, 1]

            loss_bif = criterion(output_bif, gt_bif)
            loss_vessel = criterion(output_vessel, gt_vessel)

            lambda1 = 0.5
            loss = lambda1 * loss_bif + (1-lambda1) * loss_vessel
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss
            train_bif_loss += loss_bif
            train_vessel_loss += loss_vessel

        train_loss /= len(train_loader)
        train_bif_loss /= len(train_loader)
        train_vessel_loss /= len(train_loader)
        print('Train avg batch loss, bif_loss, vessel_loss:', train_loss.item(), train_bif_loss.item(), train_vessel_loss.item())
        print('\n')
        train_stat_loss.append([train_loss.item(), train_bif_loss.item(), train_vessel_loss.item()])


if __name__ == '__main__':
    args = parse_args()
    main(args)