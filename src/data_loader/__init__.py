from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from .mnist import CustomizedMNIST
from .cifar import CustomizedCIFAR10


__all__ = ['get_dataset', 'get_dataloader']


def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if cfg.dataset == 'mnist':
        return CustomizedMNIST(
            cfg.dataset_path.mnist,
            mode=mode,
            filtering_class=cfg.mnist.in_class,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )
    if cfg.dataset == 'celeba':
        if mode == 'val':
            val = 'valid'
        return torchvision.datasets.CelebA(
            cfg.dataset_path.celeba,
            split=mode,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )
    if cfg.dataset == 'cifar10':
        if mode == 'val':
            val = 'valid'
        return CustomizedCIFAR10(
            cfg.dataset_path.cifar10,
            mode=mode,
            filtering_class=cfg.cifar10.in_class,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )

    raise ValueError('Unrecognized dataset name.')


def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']

    batch_size = getattr(cfg, mode).batch_size
    shuffle = True if mode == 'train' else False
    num_workers = getattr(cfg, mode).num_workers

    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)

    return dataloader
