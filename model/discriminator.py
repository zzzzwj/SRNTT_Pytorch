import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, act):
        super(ConvBlock, self).__init__()
        self.act = act
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        output = self.act(self.bn2(self.conv2(x)))
        return output


class Discriminator(nn.Module):
    def __init__(self, input_size, df_dim=32):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        layers = []
        in_channel = 3
        for i in range(5):
            out_channel = df_dim * 2 ** i
            layers.append(ConvBlock(in_channel, out_channel, act))
            in_channel = out_channel
        size = np.prod(input_size, axis=0)
        self.convs = nn.Sequential(*layers)
        self.fc1 = nn.Sequential(nn.Linear(size, 1024), act)
        self.fc2 = nn.Linear(1024, 1)

        self._init_param()

    def _init_param(self):
        def _norm_init_conv2d(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)

        def _norm_init_batchnorm(m):
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1., std=0.02)

        self.convs.apply(_norm_init_conv2d)
        self.convs.apply(_norm_init_batchnorm)

    def forward(self, x):
        x = self.convs(x)
        x = self.fc1(x)
        logits = self.fc2(x)
        output = torch.sigmoid(logits)
        return output, logits
