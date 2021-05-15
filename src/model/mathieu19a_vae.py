import torch
import torch.utils.data
from torch import nn, optim
from torch.distributions import Normal, Laplace, Cauchy, kl_divergence
from torch.nn import functional as F

from . import arch


class Sparse(dist.Distribution):
    has_rsample = False

    @property
    def mean(self):
        return (1 - self.gamma) * self.loc

    @property
    def stddev(self):
        return self.gamma * self.alpha + (1 - self.gamma) * self.scale

    def __init__(self, gamma, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.gamma = gamma
        self.alpha = torch.tensor(0.05).to(self.loc.device)
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        super(Sparse, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.bernoulli(self.gamma * torch.ones(shape).to(self.loc.device))
        res = p * self.alpha * torch.randn(shape).to(self.loc.device) + \
            (1 - p) * (self.loc + self.scale * torch.randn(shape).to(self.loc.device))
        return res

    def log_prob(self, value):
        res = torch.cat([(dist.Normal(torch.zeros_like(self.loc), self.alpha).log_prob(value) + self.gamma.log()).unsqueeze(0),
                         (dist.Normal(self.loc, self.scale).log_prob(value) + (1 - self.gamma).log()).unsqueeze(0)],
                        dim=0)
        return torch.logsumexp(res, 0)


class BaseVAE(nn.Module):

    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.cfg = cfg

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1) * self.cfg.mathieu.prior_std)

        self.enc = arch.enc_conv_in28out256_v1()
        self.dec = arch.dec_deconv_out28_v1(self.cfg.mathieu.z_dim)

        self.fc21 = nn.Linear(256, self.cfg.mathieu.z_dim)
        self.fc22 = nn.Linear(256, self.cfg.mathieu.z_dim)

    @property
    def z_prior(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        x = self.dec(z.view(z.size(0), -1, 1, 1))
        return torch.sigmoid(x)

    def likelihood(self, x, x_p):
        return Normal(x_p, self.cfg.mathieu.recon_std).log_prob(x)

    def forward(self, x, global_step=0):
        # B x H x W
        B, C, H, W = x.shape

        z, z_posterior = self.encode(x)
        print('z.shape', z.shape)

        x_recon = self.decode(z)

        kl = self.calc_kl_divergence(
            z_posterior, self.z_prior).flatten(start_dim=1).sum(1)

        #reg = regs(
        #    pz.sample(torch.Size([x.size(0)])).view(-1, z.size(-1)),
        #    z.squeeze(0))
        reg = 0

        log_like = self.likelihood(x, x_recon).flatten(start_dim=1).sum(1)
        elbo = log_like - self.cfg.mathieu.beta * kl \
                - self.cfg.mathieu.alpha * reg
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

    def calc_kl_divergence(self, posterior, prior, samples=None):
        if (type(posterior), type(prior)) in torch.distributions.kl._KL_REGISTRY:
            return dist.kl_divergence(p, q)
        if samples is None:
            K = 10
            samples = p.rsample(torch.Size([K])) if p.has_rsample \
                else p.sample(torch.Size([K]))
            print('samples shape', samples.shape)
        return (p.log_prob(samples) - q.log_prob(samples)).mean(0) # simple MC estimator


class SparseVAE(BaseVAE):

    @property
    def z_prior(self):
        return Sparse(
            self.cfg.mathieu.gamma,
            self.cfg.mathieu.loc,
            self.cfg.mathieu.scale)

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.enc(x)).view(x.size(0), -1)
        location, scale = self.fc21(h1), self.fc22(h1)
        scale = F.softplus(scale)
        z_posterior = Normal(location, scale)
        z = z_posterior.rsample()
        return z, z_posterior
