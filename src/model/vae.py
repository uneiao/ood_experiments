import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions import Normal, Laplace, Cauchy, kl_divergence
from torch.nn import functional as F


class BaseVAE(nn.Module):

    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.cfg = cfg

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.cfg.model.z_dim)
        self.fc22 = nn.Linear(400, self.cfg.model.z_dim)
        self.fc3 = nn.Linear(self.cfg.model.z_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    @property
    def z_prior(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def likelihood(self, x, x_p):
        return Normal(x_p, self.cfg.model.recon_std).log_prob(x)

    def forward(self, x, global_steps):
        # B x H x W
        B, C, H, W = x.shape

        z, z_posterior = self.encode(x.view(B, -1))

        x_recon = self.decode(z)

        kl = kl_divergence(z_posterior, self.z_prior)

        log_like = self.likelihood(x.view(B, -1), x_recon)
        elbo = log_like - kl
        loss = -elbo

        log = {
            'kl': kl.flatten(start_dim=1).sum(dim=1),
            'log_like': log_like.flatten(start_dim=1).sum(dim=1),
            'imgs': x,
            'y': x_recon.view(B, C, H, W),
            'loss': loss,
            'elbo': elbo,
        }

        return loss, log


class LaplaceVAE(BaseVAE):

    @property
    def z_prior(self):
        return Laplace(self.prior_mean, self.prior_std)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        location, scale = self.fc21(h1), self.fc22(h1)
        scale = F.softplus(scale)
        z_posterior = Laplace(location, scale)
        z = z_posterior.rsample()
        return z, z_posterior
