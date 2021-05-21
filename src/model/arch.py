import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F




def enc_conv_in28out256_v1(in_channel=1):
    enc = nn.Sequential(
        nn.Conv2d(in_channel, 16, 4, 2, 1),
        nn.CELU(),
        nn.GroupNorm(4, 16),
        nn.Conv2d(16, 64, 4, 2, 1),
        nn.CELU(),
        nn.GroupNorm(8, 64),
        nn.Conv2d(64, 128, 3, 2),
        nn.CELU(),
        nn.GroupNorm(16, 128),
        nn.Conv2d(128, 256, 3, 1),
        nn.CELU(),
        nn.GroupNorm(32, 256),
    )
    return enc


def enc_conv_in32out256_v1(in_channel=1):
    enc = nn.Sequential(
        nn.Conv2d(in_channel, 16, 4, 2, 1),
        nn.CELU(),
        nn.GroupNorm(4, 16),
        nn.Conv2d(16, 64, 4, 2, 1),
        nn.CELU(),
        nn.GroupNorm(8, 64),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.CELU(),
        nn.GroupNorm(16, 128),
        nn.Conv2d(128, 256, 3, 1),
        nn.CELU(),
        nn.GroupNorm(32, 256),
        nn.Conv2d(256, 256, 2, 1),
        nn.CELU(),
        nn.GroupNorm(32, 256),
    )
    return enc


def enc_conv_in28out256_v2(in_channel=1):
    enc = nn.Sequential(
        nn.Conv2d(in_channel, 32, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 2),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, 1),
        nn.ReLU(),
    )
    return enc


def dec_deconv_out28_v1(z_dim, out_channel=1):
    dec = nn.Sequential(
        nn.Conv2d(z_dim, 256, 1),
        nn.CELU(),
        nn.GroupNorm(16, 256),

        nn.Conv2d(256, 128 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(16, 128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(16, 128),

        nn.Conv2d(128, 128 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(16, 128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(16, 128),

        nn.Conv2d(128, 64 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(8, 64),
        nn.Conv2d(64, 64, 2, 1),
        nn.CELU(),
        nn.GroupNorm(8, 64),

        nn.Conv2d(64, 32 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(8, 32),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(8, 32),

        nn.Conv2d(32, 16 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(4, 16),
        nn.Conv2d(16, 16, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(4, 16),

        nn.Conv2d(16, out_channel, 3, 1, 1),
    )
    return dec


def dec_deconv_out32_v1(z_dim, out_channel=1):
    dec = nn.Sequential(
        nn.Conv2d(z_dim, 256, 1),
        nn.CELU(),
        nn.GroupNorm(16, 256),

        nn.Conv2d(256, 128 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(16, 128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(16, 128),

        nn.Conv2d(128, 128 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(16, 128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(16, 128),

        nn.Conv2d(128, 64 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(8, 64),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(8, 64),

        nn.Conv2d(64, 32 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(8, 32),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(8, 32),

        nn.Conv2d(32, 16 * 2 * 2, 1),
        nn.PixelShuffle(2),
        nn.CELU(),
        nn.GroupNorm(4, 16),
        nn.Conv2d(16, 16, 3, 1, 1),
        nn.CELU(),
        nn.GroupNorm(4, 16),

        nn.Conv2d(16, out_channel, 3, 1, 1),
    )
    return dec


def dec_deconv_out28_v2(z_dim, out_channel=1):
    dec = nn.Sequential(
        nn.ConvTranspose2d(z_dim, 128, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, out_channel, kernel_size=4, stride=2, padding=1),
    )
    return dec


MODULES = {m.__name__: m for m in [
    enc_conv_in28out256_v1,
    enc_conv_in32out256_v1,
    enc_conv_in28out256_v2,
    dec_deconv_out28_v1,
    dec_deconv_out32_v1,
    dec_deconv_out28_v2,
]}
