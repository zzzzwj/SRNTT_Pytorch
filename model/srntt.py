import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, act):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.act = act

    def forward(self, x):
        output = self.act(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = x + output
        return output


class SRNTT(nn.Module):
    def __init__(self, n_resblocks, use_weights=False, concat=False):
        super(SRNTT, self).__init__()
        self.n_resblocks = n_resblocks
        self.use_weights = use_weights
        self.concat = concat

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.content_extractor = self._content_extractor()
        self.reconstructor = self._reconstructor()

        if self.use_weights and self.concat:
            self.a = torch.ones(3)
            self.a.requires_grad = True
            self.b = torch.zeros(3)
            self.b.requires_grad = True
        self.texture_transfer = self._texture_transfer()
        self.texture_fusion_medium = self._texture_fusion('medium')
        self.texture_fusion_large = self._texture_fusion('large')
        self.srntt_out = nn.Sequential(nn.Conv2d(32, 3, 1, 1, 0), self.tanh)

        self._init_param()

    def _init_param(self):
        def _norm_init_conv2d_(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.02)

        def _norm_init_batchnorm_(m):
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1., std=0.02)

        self.content_extractor.apply(_norm_init_conv2d_)
        self.content_extractor.apply(_norm_init_batchnorm_)
        self.reconstructor.apply(_norm_init_conv2d_)
        self.texture_transfer.apply(_norm_init_conv2d_)
        self.texture_transfer.apply(_norm_init_batchnorm_)
        self.texture_fusion_medium.apply(_norm_init_conv2d_)
        self.texture_fusion_medium.apply(_norm_init_batchnorm_)
        self.texture_fusion_large.apply(_norm_init_conv2d_)
        self.texture_fusion_large.apply(_norm_init_batchnorm_)
        self.srntt_out.apply(_norm_init_conv2d_)

    def _content_extractor(self):
        # (3, h, w) => (64, h, w)
        layers = [nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), self.relu)]

        # (64, h, w) => (64, h, w)
        res_layers = []
        for _ in range(16):
            res_layers.append(ResBlock(64, 64, self.relu))
        layers.append(nn.Sequential(*res_layers))

        # (64, h, w) => (64, h, w)
        layers.append(nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        ))
        return nn.ModuleList(layers)

    def _reconstructor(self):
        layers = [
            # (64, h, w) => (64, h*2, w*2)
            nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(upscale_factor=2), self.relu),
            # (64, h*2, w*2) => (64, h*4, w*4)
            nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(upscale_factor=2), self.relu),
            # (64, h*4, w*4) => (3, h*4, w*4)
            nn.Sequential(nn.Conv2d(64, 3, 1, 1, 0), self.tanh),
        ]
        return nn.Sequential(*layers)

    def _texture_transfer(self):
        layers = [nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), self.relu)]

        res_blocks = []
        for _ in range(self.n_resblocks):
            res_blocks.append(ResBlock(64, 64, self.relu))
        layers.append(nn.Sequential(*res_blocks))

        layers.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64)))
        layers.append(nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(upscale_factor=2), self.relu))

        return nn.ModuleList(layers)

    def _texture_fusion(self, name):
        if name == 'medium':
            n_resblocks = self.n_resblocks // 2
            end_layer = nn.Sequential(nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(upscale_factor=2), self.relu)
        elif name == 'large':
            n_resblocks = self.n_resblocks // 4
            end_layer = nn.Conv2d(64, 32, 3, 1, 1)
        else:
            raise ValueError('No such layer name {}!'.format(name))

        layers = [nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), self.relu)]
        res_blocks = []
        for _ in range(n_resblocks):
            res_blocks.append(ResBlock(64, 64, self.relu))
        layers.append(nn.Sequential(*res_blocks))

        layers.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64)))
        layers.append(end_layer)

        return nn.ModuleList(layers)

    def forward(self, x, weights, maps):
        x = self.content_extractor[0](x)
        output = self.content_extractor[1](x)
        content_feature = self.content_extractor[2](output)
        output = x + content_feature
        output_upscale = self.reconstructor(output)

        if maps is None:
            return output_upscale, None

        assert isinstance(maps, (list, tuple))

        output = content_feature
        for idx, sub_module in enumerate([self.texture_transfer, self.texture_fusion_medium, self.texture_fusion_large]):
            if self.use_weights and self.concat:
                new_weights = F.interpolate(weights, scale_factor=2**idx, mode='bicubic', align_corners=True)
                map_ref = maps[idx] * torch.sigmoid(self.a[idx] * new_weights + self.b[idx])
            else:
                map_ref = maps[idx]
            map_in = output
            output = sub_module[0](torch.cat([map_in, map_ref], dim=1))
            output = sub_module[1](output)
            output = sub_module[2](output)
            output = sub_module[3](output + map_in)
        output_srntt = self.srntt_out(output)

        return output_upscale, output_srntt


if __name__ == '__main__':
    net = SRNTT(8, True, True)
    x = torch.randn(1, 3, 16, 16)
    maps = [torch.randn(1, 64, 16, 16), torch.randn(1, 64, 32, 32), torch.randn(1, 64, 64, 64)]
    weights = torch.randn(1, 1, 16, 16)
    net(x, weights, maps)
