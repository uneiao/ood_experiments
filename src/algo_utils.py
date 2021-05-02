import torch
import torch.nn.functional as F


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


def kl_divergence_bern_bern(prob_p1, prob_p2, eps=1e-15):
    """
    Compute kl divergence of two Bernoulli distributions
    :param prob_p1
    :param prob_p2
    :return: kl divergence
    """
    kl = prob_b1 * (torch.log(prob_b1 + eps) - torch.log(prob_p2 + eps)) + \
         (1 - prob_b1) * (torch.log(1 - prob_b1 + eps) - torch.log(1 - prob_p2 + eps))

    return kl


def kl_divergence_spike_slab(normal_post, normal_prior, spike_post, spike_prior):
    spike_part = (1 - spike_post).mul(torch.log((1 - spike_post) \
        / (1 - spike_prior))) \
        + spike_post.mul(torch.log(spike_post / spike_prior))

    return spike_post * kl_divergence(normal_post, normal_prior) + spike_part
