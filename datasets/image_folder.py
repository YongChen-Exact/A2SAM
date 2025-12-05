import os
import json
from PIL import Image

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register


@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', is_npy=False, mask=False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key
        self.size = size
        self.mask = mask
        self.is_npy = is_npy
        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = [] # 含根路径的文件名

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)
    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        if self.is_npy:
            return self.npy_process(x)
        if self.cache == 'none':
            return self.img_process(x)
        elif self.cache == 'in_memory':
            return x

    def npy_process(self, file):
        f = np.load(file)
        if f.ndim == 3:
            f = np.squeeze(f, axis=0)
        return f

    def img_process(self, file):
        if self.mask:
            return Image.open(file).convert('L')
        else:
            return Image.open(file).convert('RGB')


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, root_path_4, **kwargs):# img, gt, uncertainty, logit
        self.names = sorted(os.listdir(root_path_1))
        self.dataset_img = ImageFolder(root_path_1, **kwargs)
        self.dataset_gt = ImageFolder(root_path_2, **kwargs, mask=True)
        self.dataset_unc = ImageFolder(root_path_3, **kwargs, is_npy=True)
        self.dataset_lgt = ImageFolder(root_path_4, **kwargs, is_npy=True)

    def __len__(self):
        return len(self.dataset_img)

    def __getitem__(self, idx):
        return self.names[idx], self.dataset_img[idx], self.dataset_gt[idx], self.dataset_unc[idx], self.dataset_lgt[idx]
