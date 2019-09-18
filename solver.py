from model.srntt import SRNTT
from model.discriminator import Discriminator
from model.vgg import VGG19


class RefSRSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.srntt = SRNTT(cfg['model']['n_resblocks'], cfg['schedule']['use_weights'], cfg['schedule']['concat'])
        self.discriminator = Discriminator(cfg['data']['input_size'])
        self.vgg19 = VGG19(cfg['model']['final_layer'], cfg['model']['prev_layer'], True)

    def train(self):
        pass
