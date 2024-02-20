import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def dec2bin(x, length):
    mask = 2 ** torch.arange(length - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).int()


def bin2dec(b, length):
    mask = 2 ** torch.arange(length - 1, -1, -1).to(b.device, torch.int)
    return torch.sum(mask * b.int(), -1)


def gen_all_binary_vectors(length: int) -> torch.Tensor:
    return (torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


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
    # solve the linear system (O O^T + lambda I) dtheta = F by Cholesky decomposition
    # the computation Q is inlined
    N, _ = O_mat.size()
    W = O_mat @ O_mat.T + lambd * torch.eye(N, device=O_mat.device)
    L = torch.linalg.cholesky(W)
    QTQv = O_mat.T @ torch.cholesky_solve(O_mat, L) @ F_vec

    return (F_vec - QTQv) / lambd
