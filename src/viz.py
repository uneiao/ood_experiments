from attrdict import AttrDict
import imageio
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid


def get_vislogger(config):
    return VizLog()


class VizLog:

    @torch.no_grad()
    def train_vis(self, writer: SummaryWriter, log, global_step, mode, num_batch=10):
        B = num_batch

        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
                if isinstance(log[key], torch.Tensor) and log[key].ndim > 0:
                    log[key] = log[key][:num_batch]
        log = AttrDict(log)

        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        recon = log.y[:, None]

        grid_image = make_grid(log.imgs, 5, normalize=False, pad_value=1)
        writer.add_image('{}/1-image'.format(mode), grid_image, global_step)

        grid_image = make_grid(log.y, 5, normalize=False, pad_value=1)
        writer.add_image('{}/2-reconstruction'.format(mode), grid_image, global_step)

        mse = (log.y - log.imgs) ** 2
        mse = mse.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        log_like, kl = (
            log['log_like'].mean(), log['kl'].mean()
        )
        loss = log.loss.mean()

        writer.add_scalar('{}/mse'.format(mode), mse.item(), global_step=global_step)
        writer.add_scalar('{}/loss'.format(mode), loss, global_step=global_step)
        writer.add_scalar('{}/log_like'.format(mode), log_like.item(), global_step=global_step)
        writer.add_scalar('{}/KL'.format(mode), kl.item(), global_step=global_step)
        if 'z_slab_mean' in log:
            writer.add_scalar('{}/z_slab_mean'.format(mode), log.z_slab_mean.mean().item(), global_step=global_step)
            writer.add_scalar('{}/z_slab_std'.format(mode), log.z_slab_std.mean().item(), global_step=global_step)

