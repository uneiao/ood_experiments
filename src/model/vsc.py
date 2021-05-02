import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions import Normal, Laplace, Cauchy, kl_divergence
from torch.nn import functional as F

from algo_utils import linear_annealing, kl_divergence_bern_bern, \
    kl_divergence_spike_slab


class VSC(nn.Module):

    def __init__(self, cfg):
        self.cfg = cfg

        self.register_buffer('prior_slab_mean', torch.zeros(1))
        self.register_buffer('prior_slab_std', torch.ones(1))

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.cfg.vsc.z_dim)
        self.fc22 = nn.Linear(400, self.cfg.vsc.z_dim)
        self.fc3 = nn.Linear(self.cfg.vsc.z_dim, 400)
        self.fc4 = nn.Linear(400, 784)

        self.fc23 = nn.Linear(400, self.cfg.vsc.z_dim)

        self.spike_mode = 'tonolini_original'
        self.register_buffer(
            'tonolini_spike_c',
            torch.tensor(self.cfg.vsc.tonolini_spike_c_start_value))

        # fixed prior
        self.register_buffer(
            'prior_spike_prob', torch.tensor(self.cfg.vsc.prior_spike_prob))

    def anneal(self, global_step):
        self.tonolini_spike_c = linear_annealing(
            self.tonolini_spike_c.device, global_step,
            self.cfg.vsc.tonolini_spike_c_start_step,
            self.cfg.vsc.tonolini_spike_c_end_step,
            self.cfg.vsc.tonolini_spike_c_start_value,
            self.cfg.vsc.tonolini_spike_c_end_value)

    @property
    def z_slab_prior(self):
        return Normal(self.prior_slab_mean, self.prior_slab_std)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))

        location, scale = self.fc21(h1), self.fc22(h1)
        scale = F.softplus(scale)
        z_slab_posterior = Normal(location, scale)
        z_slab = z_slab_posterior.rsample()

        # compute the spike variables.
        if self.spike_mode == 'tonolini_original':
            # 1. inferring spike variable from xi.
            # 2. why not just use gumbel softmax?
            spike = torch.exp(-F.relu(self.fc23(h1))) # notice here the spike is the actual Bernoulli prob
            eta = torch.rand_like(spike)
            # ugly relaxations
            sampled_gamma = F.sigmoid(self.tonolini_spike_c * (eta + spike - 1)) # also called as 'selection'
            spike_posterior = spike

        z = z_slab * sampled_gamma
        return z, z_slab_posterior, spike_posterior

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def likelihood(self, x, x_p):
        return Normal(x_p, self.cfg.vae.recon_std).log_prob(x)

    def forward(self, x, global_step=0):
        self.annealing(golbal_step)
        # B x H x W
        B, C, H, W = x.shape

        z, z_slab_posterior, spike_posterior = self.encode(x.view(B, -1))

        x_recon = self.decode(z)

        kl1 = kl_divergence_spike_slab(
            z_slab_posterior, self.z_slab_prior, spike_posterior, self.prior_spike_prob)
        kl2 = kl_divergence_bern_bern(spike_posterior, self.prior_spike_prob)
        kl = kl1 + kl2

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
            'z': z,
        }

        return loss, log
