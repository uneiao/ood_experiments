import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions import Normal, Laplace, Cauchy, kl_divergence
from torch.nn import functional as F


class BaseVAE(nn.Module):
    def __init__(self, cfg):
        super(BaseVAE, self).__init__()

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

    def likelihood(x, x_p):
        return Normal(x_p, self.cfg.model.recon_std).logprob(x)

    def forward(self, x):
        z, z_posterior = self.encode(x)

        x_recon = self.decode(z)

        kl = kl_divergence(z_posterior, self.z_prior)

        log_like = self.likelihood(x, x_recon)
        elbo = log_like - kl
        loss = -elbo.mean()

        log = {
            'kl': kl.flatten(start_dim=1).sum(dim=1),
            'log_like': log_like.flatten(start_dim=1).sum(dim=1),
            'imgs': x,
            'y': x_recon,
            'loss': loss,
        }

        return x_recon, kl, elbo, log


class LaplaceVAE(nn.Module):

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
