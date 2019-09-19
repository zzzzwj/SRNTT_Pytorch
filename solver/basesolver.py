import os
import time
import matplotlib.pyplot as plt

from utils import utils
from data import getDataSet
from utils.config import save_config

import torch
from torch.utils.data import DataLoader


class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        train_dataset = getDataSet(cfg['data']['train'], 'train', cfg['data']['scale'])
        self.train_loader = DataLoader(train_dataset, cfg['data']['train']['batch_size'], shuffle=True,
                                       num_workers=cfg['data']['train']['n_workers'])
        val_dataset = getDataSet(cfg['data']['val'], 'val', cfg['data']['scale'])
        self.val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=cfg['data']['val']['n_workers'])
        self.records = {'Epoch': [], 'PSNR': [], 'SSIM': []}
        self.log_dir = os.path.join(cfg['output_dir'], cfg['name'],
                                    time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
        self.logger = utils.Logger(os.path.join(self.log_dir, 'info.log'))
        self.max_epochs = cfg['schedule']['num_epochs']
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoint')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.epoch = 1
        save_config(cfg, os.path.join(self.log_dir, 'config.yml'))

        self.logger.log('Train dataset has {} images and {} batches.'.format(len(train_dataset), len(self.train_loader)))
        self.logger.log('Val dataset has {} images and {} batches.'.format(len(val_dataset), len(self.val_loader)))

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_records(self):
        with open(os.path.join(self.log_dir, 'records.txt'), 'wt') as f:
            for i in range(len(self.records['Epoch'])):
                f.write('Epoch {:03d}: PSNR={:.8f}, SSIM={:.8f}\n'.format(self.records['Epoch'][i],
                                                                          self.records['PSNR'][i],
                                                                          self.records['SSIM'][i]))
        plt.figure()
        plt.plot(self.records['Epoch'], self.records['PSNR'])
        plt.savefig(os.path.join(self.log_dir, 'PSNRCurve.pdf'))

        plt.figure()
        plt.plot(self.records['Epoch'], self.records['SSIM'])
        plt.savefig(os.path.join(self.log_dir, 'SSIMCurve.pdf'))

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def run(self):
        while self.epoch <= self.max_epochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            self.save_records()
            self.epoch += 1
