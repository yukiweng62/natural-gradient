import sys

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

from model import GRU, NADE, SimpleMADE
from utils import gen_all_binary_vectors

torch.manual_seed(0)


# define model
n = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
kwargs_dict = {"n": n, "device": device, "hidden_dim": 8}
model = SimpleMADE(**kwargs_dict)
# model = NADE(**kwargs_dict)
# model = GRU(**kwargs_dict)

# check normalization
model = model.to(device)
if n <= 12:
    all_configs = gen_all_binary_vectors(n).float().to(device)
    log_probs = model(all_configs)
    assert torch.allclose(log_probs.exp().sum().cpu(), torch.tensor([1.0]))


# per sample grads, naive implementation
def loss_fn(log_probs):
    return log_probs.mean(0)


def compute_grad(sample):
    sample = sample.unsqueeze(0)
    log_probs = model(sample)
    loss = loss_fn(log_probs)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads_naive(samples):
    sample_grads = [compute_grad(samples[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


samples = model.sample(batch_size)
per_sample_grads = compute_sample_grads_naive(samples)
# print(per_sample_grads)

"""
# per sample grad using vmap
def compute_loss(params, sample):
    batch = sample.unsqueeze(0)
    log_prob = functional_call(model, (params,), (batch,))
    loss = loss_fn(log_prob)
    return loss


params = {k: v.detach() for k, v in model.named_parameters()}
ft_compute_grad = grad(compute_loss)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
ft_per_sample_grads = ft_compute_sample_grad(params, samples)
# print(ft_per_sample_grads)
"""

# per sample grad using vmap
ft_per_sample_grads = model.per_sample_grad(samples)

# check per sample grads
for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads.values()):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=3e-3, rtol=1e-5)

# benchmark
without_vmap = benchmark.Timer(
    stmt="compute_sample_grads_naive(samples)",
    setup="from __main__ import compute_sample_grads_naive",
    globals={"samples": samples},
)

with_vmap = benchmark.Timer(
    stmt="model.per_sample_grad(samples)",
    setup="from __main__ import model",
    globals={"samples": samples},
)

print(without_vmap.timeit(10))
print(with_vmap.timeit(10))
