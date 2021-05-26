import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import *


class DQN_dila(nn.Module):

    def __init__(self, cfgs):
        super(DQN_dila, self).__init__()

        self.dueling_dqn = cfgs.dueling_dqn
        input_channel = cfgs.input_channel
        init_channels = cfgs.init_channels
        output_num = cfgs.output_channel

        self.features = nn.Sequential(
            nn.Conv3d(input_channel, init_channels, kernel_size=3),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels, kernel_size=3),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels, kernel_size=3, dilation=4),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels*2, kernel_size=3),
            nn.BatchNorm3d(init_channels*2),
            nn.ReLU(),

            nn.Conv3d(init_channels*2, init_channels*2, kernel_size=1),
            nn.BatchNorm3d(init_channels*2),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Conv3d(init_channels*2, output_num, kernel_size=1),
        )

        if self.dueling_dqn:
            self.value = nn.Sequential(
                nn.Conv3d(init_channels*2, 1, kernel_size=1),
            )

    def forward(self, x):
        if self.dueling_dqn:
            x = self.features(x)
            advantage = self.advantage(x)
            value = self.value(x)
            output = value + advantage - advantage.mean()
            output = output.squeeze(-1).squeeze(-1).squeeze(-1)
            return output

        else:
            x = self.features(x)
            output = self.advantage(x)
            output = output.squeeze(-1).squeeze(-1).squeeze(-1)
            return output


class Classify_Dila(nn.Module):

    def __init__(self, cfgs):
        super(Classify_Dila, self).__init__()
        input_channel = cfgs.input_channel
        init_channels = cfgs.init_channels
        output_num = cfgs.output_num_dis

        self.features = nn.Sequential(
            nn.Conv3d(input_channel, init_channels, kernel_size=3),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels, kernel_size=3),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels, kernel_size=3, dilation=2),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels, kernel_size=3, dilation=4),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(),

            nn.Conv3d(init_channels, init_channels*2, kernel_size=3),
            nn.BatchNorm3d(init_channels*2),
            nn.ReLU(),

            nn.Conv3d(init_channels*2, init_channels*2, kernel_size=1),
            nn.BatchNorm3d(init_channels*2),
            nn.ReLU(),

            nn.Conv3d(init_channels*2, output_num, kernel_size=1),
        )


    def forward(self, x):
        output = self.features(x)
        output = output.squeeze(-1).squeeze(-1).squeeze(-1)
        return output



def optimize_model(device, policy_net, target_net, optimizer, scheduler, memory, 
                   batch_size, gamma, double_dqn=True, prioritized_replay=True):
    policy_net.train()
    target_net.eval()

    if prioritized_replay:
        # state, action, reward, next_state, indices, weights = memory.sample(batch_size)
        batch, indices, weights = memory.sample(batch_size)
        weights = torch.tensor(weights).cuda()
    else:
        if len(memory) < batch_size:
            return
        
        transitions = memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)                                                                                                                                                                                                                                                                                                                                     
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device).float()


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions t  aken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = state_action_values.sum(1).unsqueeze(1)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device).float()

    if double_dqn:
        next_q_values = policy_net(non_final_next_states)
        next_q_state_values = target_net(non_final_next_states)


        index = torch.max(next_q_values, 1)[1].unsqueeze(1)
        next_state_values[non_final_mask] = next_q_state_values.gather(1, index).squeeze(1).detach()

    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    if prioritized_replay:
        prios = state_action_values - expected_state_action_values.unsqueeze(1)
        prios = prios.data.cpu().abs().numpy() + 1e-5
        memory.update_priorities(indices, prios)
        loss = F.mse_loss(torch.mul(state_action_values, weights), torch.mul(expected_state_action_values.unsqueeze(1), weights))
    else:
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    scheduler.step()
    optimizer.step()
    del loss