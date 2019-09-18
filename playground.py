import tensorflow as tf
import torch

import numpy as np
from math import ceil


def cubic(x, scale):
    assert scale <= 1
    x = np.array(x * scale).astype(np.float64)
    abs_x = np.absolute(x)
    abs_x2 = np.multiply(abs_x, abs_x)
    abs_x3 = np.multiply(abs_x2, abs_x)
    w = np.multiply(1.5*abs_x3 - 2.5*abs_x2 + 1, abs_x <= 1) + \
        np.multiply(-0.5*abs_x3 + 2.5*abs_x2 - 4*abs_x + 2, (1 < abs_x) & (abs_x <= 2))
    return w * scale


def kernel(in_length, out_length):
    # assume in_length is larger scale
    # decide whether a convolution kernel can be constructed
    assert in_length >= out_length and in_length / out_length == 1.0 * in_length / out_length

    # decide kernel width
    scale = 1.0 * out_length / in_length
    kernel_length = 4.0 / scale

    # calculate kernel weights and padding (symmetric)
    x = np.array([1, out_length]).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_length / 2)
    p = int(ceil(kernel_length)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(p) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = cubic(np.expand_dims(u, axis=1) - indices - 1, scale)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store].squeeze()
    indices = indices[:, ind2store].squeeze()
    assert (weights[0] == weights[-1]).all()
    pad_l = int(np.where(indices[0] == 0)[0])
    pad_r = len(indices[-1]) - 1 - int(np.where(indices[-1] == in_length-1)[0])

    return weights[0], [pad_l, pad_r]


def construct_tf_kernels(in_size, out_size):
    kernels, padding = [], []
    for i in range(2):
        k, p = kernel(in_size[0], out_size[0])
        k = np.expand_dims(k, axis=1-i)
        for _ in range(2):
            k = np.expand_dims(k, axis=-1)
        kernels.append(k)
        padding.append(p)
    return kernels, padding


if __name__ == '__main__':
    k, p = construct_tf_kernels((80, 80), (40, 40))
    print(type(k[0]))
    print(type(p[0][0]))
