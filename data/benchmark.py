import os
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset


class Benchmark(Dataset):
    def __init__(self, cfg, split, scale):
        self.split = split
        self.input_root = os.path.join(cfg['input_dir'], 'HR')
        self.scale = scale

        self.files_input = sorted(glob(os.path.join(self.input_root, '*.png')))

    def __len__(self):
        return len(self.files_input)

    def __getitem__(self, idx):
        img_hr = Image.open(self.files_input[idx])
        w, h = img_hr.size
        img_lr = img_hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        img_lr = torch.from_numpy(np.array(img_lr).transpose((2, 0, 1))).float()
        img_hr = torch.from_numpy(np.array(img_hr).transpose((2, 0, 1))).float()
        # normalized to [-1, 1]
        img_lr = img_lr / 127.5 - 1
        img_hr = img_hr / 127.5 - 1
        return img_lr, img_hr
