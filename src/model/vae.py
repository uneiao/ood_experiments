import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions import Normal, Laplace, Cauchy, kl_divergence
from torch.nn import functional as F

from . import arch


class BaseVAE(nn.Module):

    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.cfg = cfg

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1) * self.cfg.vae.prior_std)
        
        self.enc = arch.enc_conv_in28out256_v1() 
        self.dec = arch.dec_deconv_out28_v1(self.cfg.vae.z_dim)

        #self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, self.cfg.vae.z_dim)
        self.fc22 = nn.Linear(256, self.cfg.vae.z_dim)
        #self.fc3 = nn.Linear(self.cfg.vae.z_dim, 256)
        #self.fc4 = nn.Linear(256, 784)

    @property
    def z_prior(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        x = self.dec(z.view(z.size(0), -1, 1, 1))
        return torch.sigmoid(x)

    def likelihood(self, x, x_p):
        return Normal(x_p, self.cfg.vae.recon_std).log_prob(x)

    def forward(self, x, global_step=0):
        # B x H x W
        B, C, H, W = x.shape

        z, z_posterior = self.encode(x)

        x_recon = self.decode(z)

        kl = kl_divergence(z_posterior, self.z_prior).flatten(start_dim=1).sum(1)

        log_like = self.likelihood(x, x_recon).flatten(start_dim=1).sum(1)
        elbo = log_like - kl
        loss = -elbo

        log = {
            'kl': kl,
            'log_like': log_like,
            'imgs': x,
            'y': x_recon.view(B, C, H, W),
            'loss': loss,
            'elbo': elbo,
            'z': z,
        }

        return loss, log


class NormalVAE(BaseVAE):

    @property
    def z_prior(self):
        return Normal(self.prior_mean, self.prior_std)

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.enc(x)).view(x.size(0), -1)
        location, scale = self.fc21(h1), self.fc22(h1)
        scale = F.softplus(scale)
        z_posterior = Normal(location, scale)
        z = z_posterior.rsample()
        return z, z_posterior


class LaplaceVAE(BaseVAE):

    @property
    def z_prior(self):
        return Laplace(self.prior_mean, self.prior_std)

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.enc(x)).view(x.size(0), -1)
        location, scale = self.fc21(h1), self.fc22(h1)
        scale = F.softplus(scale)
        z_posterior = Laplace(location, scale)
        z = z_posterior.rsample()
        return z, z_posterior
