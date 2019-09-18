import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(x):
    # b, c, h, w
    _, c, h, w = x.shape
    x = x.view(-1, c, h * w)
    return torch.matmul(x, x.permute(0, 2, 1))


def square_diff(x, y):
    return (x - y) * (x - y)


class TextureLoss(nn.Module):
    def __init__(self, use_weight_map, input_size):
        super(TextureLoss, self).__init__()
        self.use_weight_map = use_weight_map
        self.input_size = input_size
        if use_weight_map:
            self.a = torch.from_numpy(np.array([-20, -20, -20]))
            self.b = torch.from_numpy(np.array([0.65, 0.65, 0.65]))

    def forward(self, sr, hr, vgg_outputs, maps, weights=None):
        if self.use_weight_map:
            assert weights is not None and len(maps) == 3

            # b, h, w => b, 1, h, w
            new_weights = weights.unsqueeze(1)
            coeff = torch.sigmoid(new_weights * self.a[0] + self.b[0])
            loss1 = square_diff(gram_matrix(maps[0] * coeff), gram_matrix(vgg_outputs[2] * coeff) / 4.
                                / (self.input_size * self.input_size * 256) ** 2).mean()

            new_weights = F.interpolate(weights.unsqueeze(1), scale_factor=2, mode='bicubic', align_corners=True)
            coeff = torch.sigmoid(new_weights * self.a[1] + self.b[1])
            loss2 = square_diff(gram_matrix(maps[1] * coeff), gram_matrix(vgg_outputs[1] * coeff) / 4.
                                / (self.input_size * self.input_size * 512) ** 2).mean()

            new_weights = F.interpolate(weights.unsqueeze(1), scale_factor=4, mode='bicubic', align_corners=True)
            coeff = torch.sigmoid(new_weights * self.a[2] + self.b[2])
            loss3 = square_diff(gram_matrix(maps[2] * coeff), gram_matrix(vgg_outputs[0] * coeff) / 4.
                                / (self.input_size * self.input_size * 1024) ** 2).mean()
        else:
            loss1 = square_diff(gram_matrix(maps[0]), gram_matrix(vgg_outputs[2]) / 4.
                                / (self.input_size * self.input_size * 256) ** 2).mean()
            loss2 = square_diff(gram_matrix(maps[1]), gram_matrix(vgg_outputs[1]) / 4.
                                / (self.input_size * self.input_size * 512) ** 2).mean()
            loss3 = square_diff(gram_matrix(maps[0]), gram_matrix(vgg_outputs[0]) / 4.
                                / (self.input_size * self.input_size * 1024) ** 2).mean()

        return (loss1 + loss2 + loss3) / 3

