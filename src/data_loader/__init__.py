from torch.utils.data import DataLoader
from torchvision import transforms

from .mnist import CustomizedMNIST


__all__ = ['get_dataset', 'get_dataloader']


def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if cfg.dataset == 'mnist':
        return CustomizedMNIST(
            cfg.dataset_path.mnist,
            mode=mode,
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
