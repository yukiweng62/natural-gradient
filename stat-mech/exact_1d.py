# Exact computation of the dynamical partition function of the 1D East model
# TODO: add FA model

import sys

sys.path.append("..")

import os

import numpy as np
import torch

from utils import bin2dec, dec2bin

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

c = 0.5  # equilibrium parameter
N = 10  # system size
t = 1000  # evolution time
dt = 0.1  # delta t
steps = int(t / dt)  # number of Trotter steps

# construct generator, where W_s = exp(-s) * K - R
K = torch.zeros(2**N, 2**N)
R = torch.zeros(2**N, 2**N)

all_configs = dec2bin(torch.arange(2**N), N)  # all possible configurations, 2^N x N
for i in range(2**N):
    escape = 0
    config = all_configs[i, :]
    kc = torch.cat((torch.ones(1), config[:-1]))  # kinetic constraint, use x_0=1
    # find neighbors of config by flipping one bit from left to right
    for j in range(N):
        config_neighbor = config.clone()
        config_neighbor[j] = 1 - config_neighbor[j]
        # Note that the i-th row and j-th column of W is W(j -> i), that is, W(config_neighbor -> config), or W_in
        # and w(x_i -> 1-x_i) = (1-c) * x_i + c * (1-x_i)
        transition_rate = (1 - c) * config_neighbor[j] + c * (1 - config_neighbor[j])
        K[i, bin2dec(config_neighbor, N)] = kc[j] * transition_rate
        escape += kc[j] * (1 - transition_rate)
    R[i, i] = escape

for logs in np.linspace(-2, 0, 21):
    path = os.path.join(f"./out/exact/1d_east_c{c:.2f}_N{N}_t{t}_dt{dt:.2f}/", f"logs{logs:.2f}/")
    if not os.path.exists(path):
        os.makedirs(path)

    s = 10**logs
    W_s = np.exp(-s) * K - R
    T = torch.eye(2**N) + W_s * dt  # T = W_s * dt + I
    p = torch.prod(c**all_configs * (1 - c) ** (1 - all_configs), dim=-1)

    T = T.to(device)
    p = p.to(device)

    logZts = 0
    for step in range(1, steps + 1):
        p_hat = T @ p
        Z = p_hat.sum()  # contribution at this step
        logZ = torch.log(Z)
        logZts += logZ
        p = p_hat / Z  # normalize
        with open(os.path.join(path, "output.txt"), "a", newline="\n") as f:
            f.write(f"{step} {logZ} {logZts}\n")
