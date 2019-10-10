import argparse
import numpy as np
from glob import glob
from os import makedirs
from scipy.misc import imread, imresize
from os.path import exists, join, split, realpath, dirname

import torch
import torch.nn.functional as F

from utils.swap import Swap
from model.vgg import VGG19
from model.srntt import SRNTT


parser = argparse.ArgumentParser('offline_patchMatch_textureSwap')
parser.add_argument('--data_folder', type=str, default='/home/zwj/Data/RefSR/DIV2K', help='The dir of dataset: CUFED or DIV2K')
args = parser.parse_args()

data_folder = args.data_folder
if 'CUFED' in data_folder:
    input_size = 40
elif 'DIV2K' in data_folder:
    input_size = 80
else:
    raise Exception('Unrecognized dataset!')

input_path = join(data_folder, 'input')
ref_path = join(data_folder, 'ref')
matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
save_path = join(data_folder, 'map_321')
if not exists(save_path):
    makedirs(save_path)

input_files = sorted(glob(join(input_path, '*.png')))
ref_files = sorted(glob(join(ref_path, '*.png')))
n_files = len(input_files)
assert n_files == len(ref_files)

srntt = SRNTT(16).cuda()
print('Loading SRNTT ...')
ckpt = torch.load('/home/zwj/Projects/Python/SRNTT_Pytorch/log/srntt_vgg19_div2k/2019-09-20-10:06:34/' +
                  'checkpoint/best.pth')
srntt.load_state_dict(ckpt['srntt'])
print('Done.')
print('Loading VGG19 ...')
net_vgg19 = VGG19('relu_5-1', ['relu_1-1', 'relu_2-1', 'relu_3-1'], True).cuda()
print('Done.')
swaper = Swap(3, 1)

print_format = '%%0%dd/%%0%dd' % (len(str(n_files)), len(str(n_files)))
for i in range(n_files):
    file_name = join(save_path, split(input_files[i])[-1].replace('.png', '.npz'))
    if exists(file_name):
        continue
    print(print_format % (i + 1, n_files))
    img_in_lr = imresize(imread(input_files[i], mode='RGB'), (input_size, input_size), interp='bicubic')
    img_in_lr = img_in_lr.astype(np.float32) / 127.5 - 1
    img_ref = imresize(imread(ref_files[i], mode='RGB'), (input_size * 4, input_size * 4), interp='bicubic')
    img_ref = img_ref.astype(np.float32) / 127.5 - 1
    img_ref_lr = imresize(img_ref, (input_size, input_size), interp='bicubic')
    img_ref_lr = img_ref_lr.astype(np.float32) / 127.5 - 1

    img_in_lr = torch.from_numpy(img_in_lr.transpose((2, 0, 1))).unsqueeze(0).cuda()
    img_ref = torch.from_numpy(img_ref.transpose((2, 0, 1))).unsqueeze(0).cuda()
    img_ref_lr = torch.from_numpy(img_ref_lr.transpose((2, 0, 1))).unsqueeze(0).cuda()

    with torch.no_grad():
        img_in_sr = (srntt(img_in_lr, None, None)[0] + 1) * 127.5
        img_ref_sr = (srntt(img_ref_lr, None, None)[0] + 1) * 127.5

        # get feature maps via VGG19
        map_in_sr = net_vgg19(img_in_sr)[0][-1]
        map_ref = net_vgg19(img_ref)[0]
        map_ref_sr = net_vgg19(img_ref_sr)[0][-1]

    # patch matching and swapping
    other_style = []
    for idx in range(len(map_ref)):
        map_ref[idx] = map_ref[idx].cpu().squeeze().numpy().transpose((1, 2, 0))
        other_style.append([map_ref[idx]])
    other_style = other_style[:-1]

    map_in_sr = map_in_sr.cpu().squeeze().numpy().transpose((1, 2, 0))
    map_ref_sr = map_ref_sr.cpu().squeeze().numpy().transpose((1, 2, 0))

    maps, weights, correspondence = swaper.conditional_swap_multi_layer(
        content=map_in_sr,
        style=[map_ref[-1]],
        condition=[map_ref_sr],
        other_styles=other_style,
        is_weight=True
    )

    # save maps
    np.savez(file_name, target_map=maps, weights=weights, correspondence=correspondence)
