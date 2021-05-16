import torch
import torch.utils.data
from torch import nn, optim
from torch import distributions
from torch.nn import functional as F
from torch.distributions.utils import broadcast_all

from . import arch


class Sparse(distributions.Distribution):
    has_rsample = False

    @property
    def mean(self):
        return (1 - self.gamma) * self.loc

    @property
    def stddev(self):
        return self.gamma * self.alpha + (1 - self.gamma) * self.scale

    def __init__(self, gamma, loc, scale, spike_scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.gamma = gamma
        self.alpha = spike_scale#torch.tensor(spike_scale).to(self.loc.device)
        if isinstance(scale, int):
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
        res = torch.cat([(distributions.Normal(torch.zeros_like(self.loc), self.alpha).log_prob(value) + self.gamma.log()).unsqueeze(0),
                         (distributions.Normal(self.loc, self.scale).log_prob(value) + (1 - self.gamma).log()).unsqueeze(0)],
                        dim=0)
        return torch.logsumexp(res, 0)


class BaseVAE(nn.Module):

    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.cfg = cfg

        self.enc = arch.enc_conv_in28out256_v1()
        self.dec = arch.dec_deconv_out28_v1(self.cfg.mathieu.z_dim)

        self.fc21 = nn.Linear(256, self.cfg.mathieu.z_dim)
        self.fc22 = nn.Linear(256, self.cfg.mathieu.z_dim)

        self.regulariser = MMD_DIM()

    @property
    def z_prior(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        x = self.dec(z.view(z.size(0), -1, 1, 1))
        return torch.sigmoid(x)

    def likelihood(self, x, x_p):
        return distributions.Normal(x_p, self.cfg.mathieu.recon_std).log_prob(x)

    def forward(self, x, global_step=0):
        # B x H x W
        B, C, H, W = x.shape

        # B x D
        z, z_posterior = self.encode(x)

        x_recon = self.decode(z)

        kl = self.calc_kl_divergence(
            z_posterior, self.z_prior).flatten(start_dim=1).sum(1)

        reg = self.regulariser(
            self.z_prior.sample(z.shape).squeeze(-1),
            z)

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
            return distributions.kl_divergence(posterior, prior)
        if samples is None:
            K = 10
            # K x B x D
            samples = posterior.rsample(torch.Size([K])) if posterior.has_rsample \
                else posterior.sample(torch.Size([K]))
        return (posterior.log_prob(samples) - prior.log_prob(samples)).mean(0) # simple MC estimator


class SparseVAE(BaseVAE):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_buffer('prior_gamma', torch.ones(1) * self.cfg.mathieu.gamma)
        self.register_buffer('prior_loc', torch.ones(1) * self.cfg.mathieu.loc)
        self.register_buffer('prior_scale', torch.ones(1) * self.cfg.mathieu.scale)
        self.register_buffer('prior_spike_scale', torch.ones(1) * self.cfg.mathieu.spike_scale)

    @property
    def z_prior(self):
        return Sparse(
            self.prior_gamma,
            self.prior_loc,
            self.prior_scale,
            self.prior_spike_scale)

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.enc(x)).view(x.size(0), -1)
        location, scale = self.fc21(h1), self.fc22(h1)
        scale = F.softplus(scale)
        z_posterior = distributions.Normal(location, scale)
        z = z_posterior.rsample()
        return z, z_posterior


class MMD_DIM:
    def __init__(self):
        self.f = unbiased_empirical_mmd_cauchy
        self.params = None
        self.samples = True
        self.name = 'mmd_dim'

    def __call__(self, i1, i2):
        """assumes will only be given two inputs (for now)"""
        return self.f(i1, i2)


def unbiased_empirical_mmd_cauchy(X, Y):
    assert X.shape == Y.shape
    batch_size, latent_dim = X.shape
    Xb = X.expand(batch_size, *X.shape)
    Yb = Y.expand(batch_size, *Y.shape)
    dists_x = (Xb - Xb.transpose(0, 1)).pow(2)
    dists_y = (Yb - Yb.transpose(0, 1)).pow(2)
    dists_c = (Xb - Yb.transpose(0, 1)).pow(2)
    stats = 0
    off_diag = 1 - torch.eye(batch_size, device=X.device)
    off_diag = off_diag.unsqueeze(-1).expand(*off_diag.shape, latent_dim)
    for scale in [.1, .2, .5, 1., 2., 5.]:
        C = 2 * scale  # 2 * latent_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)
        res1 = off_diag * res1
        res1 = res1.sum(0).sum(0) / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum(0).sum(0) * 2. / (batch_size)
        stats += (res1 - res2).sum() # see http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture5_distribEmbed.pdf page 52
    return stats / batch_size
