import torch
import torch.nn.functional as F

import numpy as np


class Swap:
    def __init__(self, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride

    def style2patches(self, feature_map=None):
        if feature_map is None:
            feature_map = self.style

        h, w, c = feature_map.shape
        # h, w, c => 1, c, h, w
        buffer = torch.from_numpy(feature_map).permute(2, 0, 1).unsqueeze(0)
        # 1, c, h, w => 1, c, k, k, L
        patches = F.unfold(buffer, self.patch_size, stride=self.stride).view(1, c, self.patch_size, self.patch_size, -1)
        # 1, c, k, k, L => 1, k, k, c, L
        return patches.permute(0, 2, 3, 1, 4).squeeze().numpy()

        # patches = []
        # for id_r in range(0, h - self.patch_size + 1, self.stride):
        #     for id_c in range(0, w - self.patch_size + 1, self.stride):
        #         patches.append(feature_map[id_r:id_r+self.patch_size, id_c:id_c+self.patch_size, :])
        # return np.stack(patches, axis=-1)

    def conditional_swap_multi_layer(self, content, style, condition, patch_size=3, stride=1, other_styles=None,
                                     is_weight=False):
        assert isinstance(content, np.ndarray)
        self.content = np.squeeze(content)
        assert self.content.ndim == 3

        assert isinstance(style, list)
        self.style = [np.squeeze(s) for s in style]
        assert all([s.ndim == 3 for s in self.style])

        assert isinstance(condition, list)
        self.condition = [np.squeeze(c) for c in condition]
        assert all([c.ndim == 3 for c in self.condition])
        assert len(self.condition) == len(self.style)

        num_channels = self.content.shape[-1]
        assert all([s.shape[-1] == num_channels for s in self.style])
        assert all([self.style[i].shape == self.condition[i].shape for i in range(len(self.style))])

        if other_styles is not None:
            assert isinstance(other_styles, list)
            assert all([isinstance(s, list) for s in other_styles])
            other_styles = [[np.squeeze(s) for s in styles] for styles in other_styles]
            assert all([all([len(s.shape) == 3 for s in styles]) for styles in other_styles])

        self.patch_size = patch_size
        self.stride = stride

        # split content, style, and condition into patches
        patches_content = self.style2patches(self.content)
        patches_style = np.concatenate(list(map(self.style2patches, self.style)), axis=-1)
        patches = np.concatenate(list(map(self.style2patches, self.condition)), axis=-1)

        # normalize content and condition patches
        norm = np.sqrt(np.sum(np.square(patches), axis=(0, 1, 2)))
        patches_style_normed = patches / norm
        norm = np.sqrt(np.sum(np.square(patches_content), axis=(0, 1, 2)))
        patches_content_normed = patches_content / norm
        del norm, patches, patches_content

        # match content and condition patches (batch-wise matching because of memory limitation)
        # the size of a batch is 512MB
        batch_size = int(1024. ** 2 * 512 / (self.content.shape[0] * self.content.shape[1]))
        num_out_channels = patches_style_normed.shape[-1]
        print('\tMatching ...')
        max_idx, max_val = None, None
        for idx in range(0, num_out_channels, batch_size):
            print('\t  Batch %02d/%02d' % (idx / batch_size + 1, np.ceil(1. * num_out_channels / batch_size)))
            batch = patches_style_normed[..., idx:idx + batch_size]
            # if self.sess:
            #     corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch}, session=self.sess)
            # else:
            #     corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch})
            batch = torch.from_numpy(batch.transpose((3, 2, 0, 1))).cuda()
            content = torch.from_numpy(content.transpose((2, 0, 1))).unsqueeze(0).cuda()
            corr = F.conv2d(content, batch, stride=(self.stride, self.stride))
            corr = corr.cpu().squeeze().numpy().transpose((1, 2, 0))

            max_idx_tmp = np.argmax(corr, axis=-1) + idx
            max_val_tmp = np.max(corr, axis=-1)
            del corr, batch

            if max_idx is None:
                max_idx, max_val = max_idx_tmp, max_val_tmp
            else:
                indices = max_val_tmp > max_val
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices]

        # compute matching similarity (inner product)
        if is_weight:
            print('\tWeighting ...')
            corr2 = np.matmul(
                np.transpose(np.reshape(patches_content_normed, (-1, patches_content_normed.shape[-1]))),
                np.reshape(patches_style_normed, (-1, patches_style_normed.shape[-1]))
            )
            weights = np.reshape(np.max(corr2, axis=-1), max_idx.shape)
            del patches_content_normed, patches_style_normed, corr2
        else:
            weights = None
            del patches_content_normed, patches_style_normed

        # stitch matches style patches according to content spacial structure
        print('\tSwapping ...')
        maps = []
        target_map = np.zeros_like(self.content)
        count_map = np.zeros(shape=target_map.shape[:2])
        for i in range(max_idx.shape[0]):
            for j in range(max_idx.shape[1]):
                target_map[i:i + self.patch_size, j:j + self.patch_size, :] += patches_style[:, :, :, max_idx[i, j]]
                count_map[i:i + self.patch_size, j:j + self.patch_size] += 1.0
        target_map = np.transpose(target_map, axes=(2, 0, 1)) / count_map
        target_map = np.transpose(target_map, axes=(1, 2, 0))
        maps.append(target_map)

        # stitch other styles
        patch_size, stride = self.patch_size, self.stride
        if other_styles:
            for style in other_styles:
                ratio = float(style[0].shape[0]) / self.style[0].shape[0]
                assert int(ratio) == ratio
                ratio = int(ratio)
                self.patch_size = patch_size * ratio
                self.stride = stride * ratio
                patches_style = np.concatenate(list(map(self.style2patches, style)), axis=-1)
                target_map = np.zeros((self.content.shape[0] * ratio, self.content.shape[1] * ratio, style[0].shape[2]))
                count_map = np.zeros(shape=target_map.shape[:2])
                for i in range(max_idx.shape[0]):
                    for j in range(max_idx.shape[1]):
                        target_map[i*ratio:i*ratio + self.patch_size, j*ratio:j*ratio + self.patch_size, :] += patches_style[:, :, :, max_idx[i, j]]
                        count_map[i*ratio:i*ratio + self.patch_size, j*ratio:j*ratio + self.patch_size] += 1.0
                target_map = np.transpose(target_map, axes=(2, 0, 1)) / count_map
                target_map = np.transpose(target_map, axes=(1, 2, 0))
                maps.append(target_map)

        return maps, weights, max_idx
