import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions import Normal, Laplace, Cauchy, kl_divergence
from torch.nn import functional as F

from algo_utils import linear_annealing, kl_divergence_bern_bern, \
    kl_divergence_spike_slab
from . import arch


class DeformVAE(nn.Module):

    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.cfg = cfg

        self.register_buffer('prior_slab_mean', torch.zeros(1))
        self.register_buffer('prior_slab_std', torch.ones(1))

        self.register_buffer('prior_warp_mean', torch.zeros(1))
        self.register_buffer('prior_warp_std', torch.ones(1))

        H, W = self.cfg.vsc.image_shape
        offset_y, offset_x = torch.meshgrid([torch.arange(H), torch.arange(W)])
        offset_y = offset_y.float() * 2 / (H - 1) - 1
        offset_x = offset_x.float() * 2 / (W - 1) - 1
        self.register_buffer(
            'meshgrid', torch.stack((offset_x, offset_y), dim=0).float())

        self.enc = arch.enc_conv_in28out256_v1()
        self.fc_what_loc = nn.Sequential(
            nn.Linear(256, 256),
            nn.CELU(),
            nn.Linear(256, self.cfg.vsc.z_dim),
        )
        self.fc_what_scale = nn.Sequential(
            nn.Linear(256, 256),
            nn.CELU(),
            nn.Linear(256, self.cfg.vsc.z_dim),
        )
        self.fc_what_spike = nn.Sequential(
            nn.Linear(256, 256),
            nn.CELU(),
            #nn.Dropout(),
            nn.Linear(256, self.cfg.vsc.z_dim),
        )

        self.fc_warp_loc = nn.Sequential(
            nn.Linear(256, 256),
            nn.CELU(),
            nn.Linear(256, self.cfg.vsc.z_dim),
        )
        self.fc_warp_scale = nn.Sequential(
            nn.Linear(256, 256),
            nn.CELU(),
            nn.Linear(256, self.cfg.vsc.z_dim),
        )

        self.dec = arch.dec_deconv_out28_v1(self.cfg.vsc.z_dim)
        self.dec_warp = arch.dec_deconv_out28_v2(self.cfg.vsc.z_dim, out_channel=2)

        self.spike_mode = 'tonolini_original'
        self.register_buffer(
            'tonolini_spike_c',
            torch.tensor(self.cfg.vsc.tonolini_spike_c_start_value))
        self.register_buffer(
            'tonolini_lambda',
            torch.tensor(self.cfg.vsc.tonolini_lambda_start_value))

        # fixed prior
        self.register_buffer(
            'prior_spike_prob', torch.tensor(self.cfg.vsc.prior_spike_prob))

    def annealing(self, global_step):
        self.tonolini_spike_c = linear_annealing(
            self.tonolini_spike_c.device, global_step,
            self.cfg.vsc.tonolini_spike_c_start_step,
            self.cfg.vsc.tonolini_spike_c_end_step,
            self.cfg.vsc.tonolini_spike_c_start_value,
            self.cfg.vsc.tonolini_spike_c_end_value)
        self.tonolini_lambda = linear_annealing(
            self.tonolini_lambda.device, global_step,
            self.cfg.vsc.tonolini_lambda_start_step,
            self.cfg.vsc.tonolini_lambda_end_step,
            self.cfg.vsc.tonolini_lambda_start_value,
            self.cfg.vsc.tonolini_lambda_end_value)

    @property
    def z_slab_prior(self):
        return Normal(self.prior_slab_mean, self.prior_slab_std)

    @property
    def z_warp_prior(self):
        return Normal(self.prior_warp_mean, self.prior_warp_std)

    def encode(self, x):
        h1 = F.relu(self.enc(x)).view(x.size(0), -1)

        z_what_location, z_what_scale = self.fc_what_loc(h1), self.fc_what_scale(h1)
        z_what_scale = F.softplus(z_what_scale)
        z_what_slab_posterior = Normal(
            self.tonolini_lambda * z_what_location,
            self.tonolini_lambda * z_what_scale - self.tonolini_lambda + 1.0)
        z_slab = z_slab_posterior.rsample()

        # compute the spike variables.
        if self.spike_mode == 'tonolini_original':
            # 1. inferring spike variable from xi.
            # 2. why not just use gumbel softmax?
            spike = torch.exp(-F.relu(self.fc_what_spike(h1))) # notice here the spike is the actual Bernoulli prob
            #spike = torch.sigmoid(self.fc23(h1)) # notice here the spike is the actual Bernoulli prob
            eta = torch.rand_like(spike)
            # ugly relaxations
            sampled_gamma = F.sigmoid(self.tonolini_spike_c * (eta + spike - 1)) # also called as 'selection'
            z_what_spike_posterior = spike

        z_what = z_slab * sampled_gamma

        z_warp_location, z_warp_scale = self.fc_warp_loc(h1), self.fc_warp_scale(h1)
        z_warp_scale = F.softplus(z_warp_scale)
        z_warp_posterior = Normal(z_warp_location, z_warp_scale)
        z_warp = z_warp_posterior.rsample()

        return z_what, z_what_slab_posterior, z_what_spike_posterior, \
                z_warp, z_warp_posterior

    def decode(self, z_what, z_warp):
        # appearance
        x = self.dec(z_what.view(z_what.size(0), -1, 1, 1))
        x = torch.sigmoid(x)

        # warping
        w = self.dec_warp(z_warp.view(z_warp.size(0), -1, 1, 1))
        flow = torch.tanh(w).permute(0, 2, 3, 1)
        grid = (flow + self.meshgrid).clamp(min=-1.0, max=1.0)
        x = F.grid_sample(x, grid)

        return x

    def likelihood(self, x, x_p):
        return Normal(x_p, self.cfg.vae.recon_std).log_prob(x)

    def forward(self, x, global_step=0):
        self.annealing(global_step)
        # B x H x W
        B, C, H, W = x.shape

        z_what, z_slab_posterior, spike_posterior, \
                z_warp, z_warp_posterior = self.encode(x)

        x_recon = self.decode(z_what, z_warp)

        kl1 = kl_divergence_spike_slab(
            z_slab_posterior, self.z_slab_prior, spike_posterior, self.prior_spike_prob)
        kl2 = kl_divergence_bern_bern(spike_posterior, self.prior_spike_prob)

        kl_warp = kl_divergence(z_warp_posterior, self.z_warp_prior)

        kl = kl1.flatten(start_dim=1).sum(1) + kl2.flatten(start_dim=1).sum(1) \
                + kl_warp.flatten(start_dim=1).sum(1)

        log_like = self.likelihood(x, x_recon).flatten(start_dim=1).sum(1)
        elbo = log_like - self.cfg.vsc.beta * kl
        loss = -elbo

        log = {
            'kl': kl,
            'log_like': log_like,
            'imgs': x,
            'y': x_recon.view(B, C, H, W),
            'loss': loss,
            'elbo': elbo,
            'z': z_what,
            'z_slab_mean': z_slab_posterior.mean,
            'z_slab_std': z_slab_posterior.stddev,
        }

        return loss, log
