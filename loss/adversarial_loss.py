import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
def sigmoid_cross_entropy_with_logits(logits, labels):
    return F.relu(logits) - logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))


class AdvLoss(nn.Module):
    def __init__(self, is_WGAN_GP, param_WGAN_GP=None):
        super(AdvLoss, self).__init__()
        self.is_WGAN_WP = is_WGAN_GP
        self.param_WGAN_GP = param_WGAN_GP

    def forward(self, srntt_out, ground_truth, d_fake_logits, d_real_logits, discriminator):
        if self.is_WGAN_WP:
            # WGAN losses
            loss_d = d_fake_logits.mean() - d_real_logits.mean()
            loss_g = -d_fake_logits.mean()

            # gradient penalty
            alpha = torch.from_numpy(np.random.uniform(0., 1., (d_fake_logits.shape[0], 1, 1, 1))).to(d_fake_logits.device)
            interpolates = alpha * ground_truth + (1. - alpha) * srntt_out
            _, disc_interpolates = discriminator(interpolates)
            # see https://stackoverflow.com/questions/49149699/differentiate-gradients
            gradients = torch.autograd.grad(disc_interpolates, interpolates)
            slopes = torch.sqrt((gradients * gradients).sum(dim=1))
            gradient_penalty = ((slopes - 1) ** 2).mean()
            loss_d = loss_d + self.param_WGAN_GP * gradient_penalty
        else:
            loss_g = sigmoid_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))
            loss_d_fake = sigmoid_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            loss_d_real = sigmoid_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits))
            loss_d = loss_d_fake + loss_d_real

        return loss_d, loss_g

