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


class CustomizedCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, *largs, **kwargs):

        mode = kwargs.pop('mode')

        split = kwargs.pop('train_val_split', [40000, 10000])

        filtering_class = kwargs.pop('filtering_class', [])

        if mode in ['train', 'val']:
            kwargs['train'] = True
        else:
            kwargs['train'] = False

        super().__init__(*largs, **kwargs)

        if mode in ['train', 'val']:
            _train, _val = getset_split_cache(
                random_split(range(len(self)), split)
            )
            indices = _train.indices if mode == 'train' else _val.indices
            self.data, self.targets = self.data[indices], np.array(self.targets)[indices]

        if mode == 'train' and len(filtering_class):
            mask = [1 if self.targets[i] in filtering_class else 0 for i in range(len(self.targets))]
            mask = torch.tensor(mask, dtype=bool)
            self.targets = self.targets[torch.nonzero(mask)].squeeze()
            self.data = self.data[torch.nonzero(mask)].squeeze()

