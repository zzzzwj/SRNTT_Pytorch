import os
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import utils
from model.srntt import SRNTT
from solver.basesolver import BaseSolver


class SISRSolver(BaseSolver):
    def __init__(self, cfg):
        super(SISRSolver, self).__init__(cfg)

        self.srntt = SRNTT(cfg['model']['n_resblocks'], cfg['schedule']['use_weights'],
                           cfg['schedule']['concat']).cuda()
        self.discriminator = None
        self.vgg = None
        params = list(self.srntt.content_extractor.parameters()) + list(self.srntt.reconstructor.parameters())
        self.num_epochs = self.cfg['schedule']['num_epochs']
        self.optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'])
        self.reconst_loss = nn.L1Loss()

    def train(self):
        with tqdm(total=len(self.train_loader), miniters=1,
                  desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.max_epochs)) as t:
            for data in self.train_loader:
                lr, hr = data['lr'].cuda(), data['hr'].cuda()
                self.srntt.train()
                self.optimizer.zero_grad()
                sr, _ = self.srntt(lr, None, None)
                loss = self.reconst_loss(sr, hr)
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                loss.backward()
                self.optimizer.step()

    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1,
                  desc='Val Epoch: [{}/{}]'.format(self.epoch, self.max_epochs)) as t:
            psnr_list, ssim_list, loss_list = [], [], []
            for lr, hr in self.val_loader:
                lr, hr = lr.cuda(), hr.cuda()
                self.srntt.eval()
                with torch.no_grad():
                    sr, _ = self.srntt(lr, None, None)
                    loss = self.reconst_loss(sr, hr)

                batch_psnr, batch_ssim = [], []
                for c in range(sr.shape[0]):
                    predict_sr = (sr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    ground_truth = (hr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    psnr = utils.calculate_psnr(predict_sr, ground_truth, 255)
                    ssim = utils.calculate_ssim(predict_sr, ground_truth, 255)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                t.set_postfix_str('Batch loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}'.format(loss.item(), avg_psnr,
                                                                                          avg_ssim))
                t.update()
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())
            self.logger.log('Val Epoch {}: PSNR={}, SSIM={}'.format(self.epoch, self.records['PSNR'][-1],
                                                                    self.records['SSIM'][-1]))

    def save_checkpoint(self):
        super(SISRSolver, self).save_checkpoint()
        self.ckp['srntt'] = self.srntt.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()

        torch.save(self.ckp, os.path.join(self.checkpoint_dir, 'latest.pth'))
        if self.records['PSNR'][-1] == np.array(self.records['PSNR']).max():
            shutil.copy(os.path.join(self.checkpoint_dir, 'latest.pth'),
                        os.path.join(self.checkpoint_dir, 'best.pth'))

    def load_checkpoint(self, model_path):
        super(SISRSolver, self).load_checkpoint(model_path)
        ckpt = torch.load(model_path)
        self.srntt.load_state_dict(ckpt['srntt'])
        self.optimizer.load_state_dict(ckpt['optimizer'])


if __name__ == '__main__':
    from utils.config import get_config
    cfg = get_config('config/srntt_sisr_only.yml')
    solver = SISRSolver(cfg)