import math

import torch


def _partial_t_prob(x, model, c, from_t0):
    kc = torch.cat(
        (torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device), x[:, :-1]), 1
    )  # kinetic constraint, (batch_size, n)
    transition_rate = (1 - c) * x + c * (1 - x)  # w(x_i -> 1-x_i) = (1-c) * x_i + c * (1-x_i)
    W_out = kc * transition_rate
    escape_rate = W_out.sum(dim=1)  # shape (batch_size)
    W_in = kc * (1 - transition_rate)  # W(x' -> x)

    if from_t0:
        # P(x) is a product state
        probs = torch.prod(c * x + (1 - c) * (1 - x), dim=1)
        probs_conn = torch.zeros(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        for i in range(x.size(1)):  # flip spins from L to R
            x_flipped = x.clone()
            x_flipped[:, i] = 1.0 - x_flipped[:, i]
            probs_conn[:, i] = torch.prod(c * x_flipped + (1 - c) * (1 - x_flipped), dim=1)
    else:
        # use the model at last step to calculate P(x)
        probs = model(x).exp()
        probs_conn = torch.zeros(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        for i in range(x.size(1)):  # flip spins from L to R
            x_flipped = x.clone()
            x_flipped[:, i] = 1.0 - x_flipped[:, i]
            probs_conn[:, i] = model(x_flipped).exp()

    return probs, probs_conn, escape_rate, W_in


def transition_prob(x, model, c, s, dt, from_t0):
    """Calculate $TP(x) = P(x) + dt * (-escape_rate * P(x) + exp(-s) * sum_{x'} P(x') * W(x' -> x))"""

    probs, probs_conn, escape_rate, W_in = _partial_t_prob(x, model, c, from_t0)
    return probs + dt * (-escape_rate * probs + math.exp(-s) * torch.sum(probs_conn * W_in, dim=1))


def partial_t_log_prob(x, model, c, s, dt, from_t0):
    """Calculate \partial_t log P(x) = -escape_rate + exp(-s) * sum_{x'} P(x') * W(x' -> x) / P(x)"""

    probs, probs_conn, escape_rate, W_in = _partial_t_prob(x, model, c, from_t0)
    return -escape_rate + math.exp(-s) * torch.sum(probs_conn * W_in, dim=1) / probs
