from skimage import io
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision


SPLIT_CACHE = None
def getset_split_cache(res):
    global SPLIT_CACHE
    if SPLIT_CACHE is None:
        SPLIT_CACHE = res
    return SPLIT_CACHE


#class CustomizedMNIST(Dataset):
class CustomizedMNIST(torchvision.datasets.MNIST):

    def __init__(self, *largs, **kwargs):

        mode = kwargs.pop('mode')

        split = kwargs.pop('train_val_split', [50000, 10000])

        if mode in ['train', 'val']:
            kwargs['train'] = True

        super().__init__(*largs, **kwargs)

        if mode in ['train', 'val']:
            _train, _val = getset_split_cache(
                random_split(range(len(self)), split)
            )
            indices = _train.indices if mode == 'train' else _val.indices
            self.data, self.targets = self.data[indices], self.targets[indices]
