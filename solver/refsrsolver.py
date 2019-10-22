import os
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from utils import utils
from model.vgg import VGG19
from model.srntt import SRNTT
from model.discriminator import Discriminator
from loss.texture_loss import TextureLoss
from loss.backprojection_loss import BackProjectionLoss
from loss.adversarial_loss import AdvLoss
from solver.basesolver import BaseSolver


class RefSRSolver(BaseSolver):
    def __init__(self, cfg):
        super(RefSRSolver, self).__init__(cfg)

        self.srntt = SRNTT(cfg['model']['n_resblocks'], cfg['schedule']['use_weights'],
                           cfg['schedule']['concat']).cuda()
        # self.discriminator = None
        self.discriminator = Discriminator(cfg['data']['input_size']).cuda()
        # self.vgg = None
        self.vgg = VGG19(cfg['model']['final_layer'], cfg['model']['prev_layer'], True).cuda()
        params = list(self.srntt.texture_transfer.parameters()) + list(self.srntt.texture_fusion_medium.parameters()) +\
                 list(self.srntt.texture_fusion_large.parameters()) + list(self.srntt.srntt_out.parameters())
        self.init_epoch = self.cfg['schedule']['init_epoch']
        self.num_epochs = self.cfg['schedule']['num_epochs']
        self.optimizer_init = torch.optim.Adam(params, lr=cfg['schedule']['lr'])
        self.optimizer = torch.optim.lr_scheduler.MultiStepLR(torch.optim.Adam(params, lr=cfg['schedule']['lr']),
                                                              [self.num_epochs // 2], 0.1)
        self.optimizer_d = torch.optim.lr_scheduler.MultiStepLR(torch.optim.Adam(self.discriminator.parameters(),
                                                                                 lr=cfg['schedule']['lr']),
                                                                [self.num_epochs // 2], 0.1)
        self.reconst_loss = nn.L1Loss()
        self.bp_loss = BackProjectionLoss()
        self.texture_loss = TextureLoss(self.cfg['schedule']['use_weights'], 80)
        self.adv_loss = AdvLoss(self.cfg['schedule']['is_WGAN_GP'])
        self.loss_weights = self.cfg['schedule']['loss_weights']

    def train(self):
        if self.epoch <= self.init_epoch:
            with tqdm(total=len(self.train_loader), miniters=1,
                      desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.max_epochs)) as t:
                for data in self.train_loader:
                    lr, hr = data['lr'].cuda(), data['hr'].cuda()
                    maps, weight = data['map'].cuda(), data['weight'].cuda()
                    self.srntt.train()
                    self.optimizer_init.zero_grad()
                    sr, srntt_out = self.srntt(lr, weight, maps)
                    loss_reconst = self.reconst_loss(sr, hr)
                    loss_bp = self.bp_loss(lr, srntt_out)
                    loss = self.loss_weights[4] * loss_reconst + self.loss_weights[3] * loss_bp
                    t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                    t.update()

                    loss.backward()
                    self.optimizer_init.step()
        elif self.epoch <= self.num_epochs:
            with tqdm(total=len(self.train_loader), miniters=1,
                      desc='Complete Training Epoch: [{}/{}]'.format(self.epoch, self.max_epochs)) as t:
                for data in self.train_loader:
                    lr, hr = data['lr'].cuda(), data['hr'].cuda()
                    maps, weight = data['map'].cuda(), data['weight'].cuda()
                    self.srntt.train()
                    self.optimizer_init.zero_grad()
                    self.optimizer.optimizer.zero_grad()
                    self.optimizer_d.optimizer.zero_grad()
                    sr, srntt_out = self.srntt(lr, weight, maps)
                    sr_prevlayer, sr_lastlayer = self.vgg(srntt_out)
                    hr_prevlayer, hr_lastlayer = self.vgg(hr)
                    _, d_real_logits = self.discriminator(hr)
                    _, d_fake_logits = self.discriminator(srntt_out)
                    loss_reconst = self.reconst_loss(sr, hr)
                    loss_bp = self.bp_loss(lr, srntt_out)
                    loss_texture = self.texture_loss(sr_prevlayer, maps, weight)
                    loss_d, loss_g = self.adv_loss(srntt_out, hr, d_fake_logits, d_real_logits, self.discriminator)
                    loss_percep = torch.pow(sr_lastlayer - hr_lastlayer, 2).mean()
                    if self.cfg['schedule']['use_lower_layers_in_per_loss']:
                        for l_sr, l_hr in zip(sr_prevlayer, hr_prevlayer):
                            loss_percep += torch.pow(l_sr - l_hr, 2).mean()
                        loss_percep = loss_percep / (len(sr_prevlayer) + 1)
                    weighted_loss = torch.Tensor(self.loss_weights).cuda() * \
                                    torch.Tensor([loss_percep, loss_texture, loss_g, loss_bp, loss_reconst])
                    total_loss = weighted_loss.sum()

                    t.set_postfix_str("Batch loss {:.4f}".format(total_loss.item()))
                    t.update()

                    loss_d.backward()
                    total_loss.backward()
                    self.optimizer.step(self.epoch)
                    self.optimizer_d.step(self.epoch)
        else:
            pass

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
        super(RefSRSolver, self).save_checkpoint()
        self.ckp['srntt'] = self.srntt.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        self.ckp['optimizer_d'] = self.optimizer_d.state_dict()
        self.ckp['optimizer_init'] = self.optimizer_init.state_dict()
        if self.discriminator is not None:
            self.ckp['discriminator'] = self.discriminator.state_dict()
        if self.vgg is not None:
            self.ckp['vgg'] = self.vgg.state_dict()

        torch.save(self.ckp, os.path.join(self.checkpoint_dir, 'latest.pth'))
        if self.records['PSNR'][-1] == np.array(self.records['PSNR']).max():
            shutil.copy(os.path.join(self.checkpoint_dir, 'latest.pth'),
                        os.path.join(self.checkpoint_dir, 'best.pth'))

    def load_checkpoint(self, model_path):
        super(RefSRSolver, self).load_checkpoint(model_path)
        ckpt = torch.load(model_path)
        self.srntt.load_state_dict(ckpt['srntt'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.optimizer_d.load_state_dict(ckpt['optimizer_d'])
        self.optimizer_init.load_state_dict(ckpt['optimizer_init'])
        if 'vgg' in ckpt.keys() and self.vgg is not None:
            self.vgg.load_stat_dict(ckpt['srntt'])
        if 'discriminator' in ckpt.keys() and self.discriminator is not None:
            self.discriminator.load_state_dict(ckpt['discriminator'])


if __name__ == '__main__':
    from utils.config import get_config
    cfg = get_config('config/srntt_vgg19_div2k.yml')
    solver = RefSRSolver(cfg)
