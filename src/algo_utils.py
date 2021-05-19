import math

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


EPS = 1e-15
SPARSE_EPS = 1e-5


def linear_annealing(device, step, start_step, end_step, start_value, end_value):
    """
    Linear annealing

    :param x: original value. Only for getting device
    :param step: current global step
    :param start_step: when to start changing value
    :param end_step: when to stop changing value
    :param start_value: initial value
    :param end_value: final value
    :return:
    """
    if step <= start_step:
        x = torch.tensor(start_value, device=device)
    elif start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step), device=device)
    else:
        x = torch.tensor(end_value, device=device)

    return x


def kl_divergence_bern_bern(prob_p1, prob_p2, eps=EPS):
    """
    Compute kl divergence of two Bernoulli distributions
    :param prob_p1
    :param prob_p2
    :return: kl divergence
    """
    kl = prob_p1 * (torch.log(prob_p1 + eps) - torch.log(prob_p2 + eps)) + \
         (1 - prob_p1) * (torch.log(1 - prob_p1 + eps) - torch.log(1 - prob_p2 + eps))

    return kl


def kl_divergence_spike_slab(normal_post, normal_prior, spike_post, spike_prior):
    spike_part = (1 - spike_post).mul(torch.log((1 - spike_post) \
        / (1 - spike_prior) + EPS)) \
        + spike_post.mul(torch.log(spike_post / spike_prior + EPS))

    return spike_post * kl_divergence(normal_post, normal_prior) + spike_part


def hoyer_metric(zs):
    '''
    see section 6.3 of http://proceedings.mlr.press/v97/mathieu19a/mathieu19a.pdf
    '''
    latent_dim = zs.size(-1)
    zs = zs / (zs.std(0) + EPS)
    l1_l2 = (zs.abs().sum(-1) / (zs.pow(2).sum(-1).sqrt() + EPS)).mean()
    return (math.sqrt(latent_dim) - l1_l2) / (math.sqrt(latent_dim) - 1)


def avg_count_sparsity(z, eps=SPARSE_EPS):
    return (z.abs() < eps).sum() / torch.ones(z.shape).sum()
