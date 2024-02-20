import sys

sys.path.append("..")

import torch
import torch.nn.functional as F

from utils import scaled_dot_product_attention

B, num_heads, T, d_k = 128, 4, 20, 16
device = "cuda" if torch.cuda.is_available() else "cpu"
Q = torch.randn(B, num_heads, T, d_k).to(device)

output1 = F.scaled_dot_product_attention(Q, Q, Q, is_causal=True)
output2 = scaled_dot_product_attention(Q, Q, Q, is_causal=True)
assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)

import timeit

t0 = timeit.Timer(
    stmt="scaled_dot_product_attention(Q, Q, Q, is_causal=True)",
    setup="from __main__ import scaled_dot_product_attention",
    globals={"Q": Q},
)

t1 = timeit.Timer(
    stmt="F.scaled_dot_product_attention(Q, Q, Q, is_causal=True)",
    setup="import torch.nn.functional as F",
    globals={"Q": Q},
)

print(f"Naive implementation: {t0.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"Fast implementation: {t1.timeit(100) / 100 * 1e6:>5.1f} us\n")

import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt="scaled_dot_product_attention(Q, Q, Q, is_causal=True)",
    setup="from __main__ import scaled_dot_product_attention",
    globals={"Q": Q},
)

t1 = benchmark.Timer(
    stmt="F.scaled_dot_product_attention(Q, Q, Q, is_causal=True)",
    setup="import torch.nn.functional as F",
    globals={"Q": Q},
)

print(t0.timeit(100))
print(t1.timeit(100))
