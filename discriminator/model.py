import torch
import torch.nn as nn


class Dilated_Regress_Net(nn.Module):

    def __init__(self, input_size, output_num, input_channel=1, start_filter_num=32):
        super(Dilated_Regress_Net, self).__init__()

        self.input_shape = (1, input_size, input_size, input_size)

        self.features = nn.Sequential(
            nn.Conv3d(input_channel, start_filter_num, kernel_size=3),
            nn.BatchNorm3d(start_filter_num),
            nn.ReLU(),

            nn.Conv3d(start_filter_num, start_filter_num, kernel_size=3),
            nn.BatchNorm3d(start_filter_num),
            nn.ReLU(),

            nn.Conv3d(start_filter_num, start_filter_num, kernel_size=3, dilation=2),
            nn.BatchNorm3d(start_filter_num),
            nn.ReLU(),

            nn.Conv3d(start_filter_num, start_filter_num, kernel_size=3, dilation=4),
            nn.BatchNorm3d(start_filter_num),
            nn.ReLU(),

            nn.Conv3d(start_filter_num, start_filter_num*2, kernel_size=3),
            nn.BatchNorm3d(start_filter_num*2),
            nn.ReLU(),

            nn.Conv3d(start_filter_num*2, start_filter_num*2, kernel_size=1),
            nn.BatchNorm3d(start_filter_num*2),
            nn.ReLU(),

            nn.Conv3d(start_filter_num*2, output_num, kernel_size=1),
        )

    def forward(self, x):
        output = self.features(x)
        return output