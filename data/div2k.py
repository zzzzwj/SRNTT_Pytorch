import os
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset


class DIV2K(Dataset):
    def __init__(self, cfg, split, scale):
        self.split = split
        self.input_root = cfg['input_dir']
        self.scale = scale
        self.map_root = cfg['map_dir']

        self.files_input = sorted(glob(os.path.join(self.input_root, '*.png')))
        self.maps = sorted(glob(os.path.join(self.map_root, )))

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
        return {'lr': img_lr, 'hr': img_hr}


class DIV2KRef(Dataset):
    def __init__(self, cfg, split, scale, use_weight):
        self.split = split
        self.input_root = cfg['input_dir']
        self.scale = scale
        self.ref_root = cfg['ref_dir']
        self.map_root = cfg['map_dir']
        self.use_weight = use_weight

        self.files_input = sorted(glob(os.path.join(self.input_root, '*.png')))
        self.files_ref = sorted(glob(os.path.join(self.ref_root, '*.png')))
        self.maps = sorted(glob(os.path.join(self.map_root, '*.npz')))
        print(len(self.maps))

        assert len(self.files_input) == len(self.files_ref) == len(self.maps)

    def __len__(self):
        return len(self.files_input)

    def __getitem__(self, idx):
        img_hr = Image.open(self.files_input[idx])
        img_ref = Image.open(self.files_ref[idx])
        zip_data = np.load(self.maps[idx], allow_pickle=True)
        # l1: 80x80, l2: 320x320, l3: 160x160
        l1_map, l2_map, l3_map = zip_data['target_map']
        w, h = img_hr.size
        img_lr = img_hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        img_lr = torch.from_numpy(np.array(img_lr).transpose((2, 0, 1))).float()
        img_hr = torch.from_numpy(np.array(img_hr).transpose((2, 0, 1))).float()
        img_ref = torch.from_numpy(np.array(img_ref).transpose((2, 0, 1))).float()
        maps = map(lambda x: torch.from_numpy(x).permute((2, 0, 1)).float(), [l1_map, l3_map, l2_map])
        # normalized to [-1, 1]
        img_lr = img_lr / 127.5 - 1
        img_hr = img_hr / 127.5 - 1

        if self.use_weight:
            weight = zip_data['weights']
            weight = np.pad(weight, ((1, 1), (1, 1)), 'edge')
        else:
            _, h, w = img_lr.shape
            weight = np.zeros(shape=(h, w))
        weight = torch.from_numpy(weight)
        return {'lr': img_lr, 'hr': img_hr, 'ref': img_ref, 'map': maps, 'weight': weight}


if __name__ == '__main__':
    from utils.config import get_config
    cfg = get_config('config/srntt_vgg19_div2k.yml')
    dataset = DIV2KRef(cfg['data']['train'], 'train', cfg['data']['scale'], use_weight=True)
    k = dataset[1]
