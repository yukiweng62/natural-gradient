import math

import torch
import torch.utils.benchmark as benchmark

torch.set_default_dtype(torch.float64)

N = 100
M = 1000
lambd = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# define O matrix (N by M) and F vector (M)
O_mat = torch.randn(N, M, device=device) / math.sqrt(N)
F_vec = torch.rand(M, device=device)


def svd_solve(O_mat, F_vec, lambd=1e-3):
    # solve the linear system (O O^T + lambda I) dtheta = F by SVD
    # compute the eigenvalue decomposition of O O^T = U Sigma^2 U^T
    Sigma2, U = torch.linalg.eigh(O_mat @ O_mat.T)
    # V = O^T U Sigma^{-1}
    V = O_mat.T @ U @ torch.diag(1.0 / torch.sqrt(Sigma2))

    return V @ torch.diag(1.0 / (Sigma2 + lambd)) @ V.T @ F_vec + (F_vec - V @ V.T @ F_vec) / lambd


def cholesky_solve(O_mat, F_vec, lambd=1e-3):
    # solve the linear system (O O^T + lambda I) dtheta = F by Cholesky decomposition
    # TODO: optimize this function
    N, _ = O_mat.size()
    W = O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device)
    L = torch.linalg.cholesky(W)
    Q = torch.linalg.inv(L) @ O_mat

    return (F_vec - Q.T @ Q @ F_vec) / lambd


def cholesky_solve_fast(O_mat, F_vec, lambd=1e-3):
    N, _ = O_mat.size()
    W = O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device)
    L = torch.linalg.cholesky(W)
    QTQv = O_mat.T @ torch.cholesky_solve(O_mat, L) @ F_vec

    return (F_vec - QTQv) / lambd


# benchmark
bench_svd_solve = benchmark.Timer(
    stmt="svd_solve(O_mat, F_vec, lambd)",
    setup="from __main__ import svd_solve",
    globals={"O_mat": O_mat, "F_vec": F_vec, "lambd": lambd},
)

bench_cholesky_solve = benchmark.Timer(
    stmt="cholesky_solve(O_mat, F_vec, lambd)",
    setup="from __main__ import cholesky_solve",
    globals={"O_mat": O_mat, "F_vec": F_vec, "lambd": lambd},
)

bench_cholesky_solve_fast = benchmark.Timer(
    stmt="cholesky_solve_fast(O_mat, F_vec, lambd)",
    setup="from __main__ import cholesky_solve_fast",
    globals={"O_mat": O_mat, "F_vec": F_vec, "lambd": lambd},
)

# print(bench_svd_solve.timeit(10))
# print(bench_cholesky_solve.timeit(10))
# print(bench_cholesky_solve_fast.timeit(10))

# t1 = bench_svd_solve.blocked_autorange()
# t2 = bench_cholesky_solve.blocked_autorange()
# t3 = bench_cholesky_solve_fast.blocked_autorange()

# print(t1)
# print(t2)
# print(t3)

results = []

for N in [10, 100, 1000]:
    for M in [10, 100, 1000, 10000]:
        # for M in [10, 100, 1000, 10000, 100000]:
        label = "Batched dot"
        sub_label = f"N={N}, M={M}"
        O_mat = torch.randn(N, M, device=device) / math.sqrt(N)
        F_vec = torch.rand(M, device=device)
        # for num_threads in [1, 4, 16, 32]:
        for num_threads in [1]:
            results.append(
                benchmark.Timer(
                    stmt="svd_solve(O_mat, F_vec, lambd)",
                    setup="from __main__ import svd_solve",
                    globals={"O_mat": O_mat, "F_vec": F_vec, "lambd": lambd},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="SVD solve",
                ).blocked_autorange(min_run_time=1)
            )
            results.append(
                benchmark.Timer(
                    stmt="cholesky_solve(O_mat, F_vec, lambd)",
                    setup="from __main__ import cholesky_solve",
                    globals={"O_mat": O_mat, "F_vec": F_vec, "lambd": lambd},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="Cholesky solve",
                ).blocked_autorange(min_run_time=1)
            )
            results.append(
                benchmark.Timer(
                    stmt="cholesky_solve_fast(O_mat, F_vec, lambd)",
                    setup="from __main__ import cholesky_solve_fast",
                    globals={"O_mat": O_mat, "F_vec": F_vec, "lambd": lambd},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description="Cholesky solve fast",
                ).blocked_autorange(min_run_time=1)
            )

compare = benchmark.Compare(results)
compare.print()
