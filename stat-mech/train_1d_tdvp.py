# TODO: TDVP not working!!!

import sys

sys.path.append("..")

import copy
import logging
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time

import numpy as np
import torch
import yaml
from args import args
from master_eq import partial_t_log_prob, transition_prob
from torch.func import functional_call, grad, vmap
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GRU, NADE, SimpleMADE
from utils import cholesky_solve


def main():
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and args.gpu > -1
    args.device = f"cuda:{args.gpu}" if use_cuda else "cpu"

    args.s = 10**args.logs
    if args.negative_s:
        args.s *= -1

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    writer = SummaryWriter(log_dir=args.path)

    # save config
    with open(os.path.join(args.path, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    logging.basicConfig(
        filename=os.path.join(args.path, "train.log"),
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        filemode="w",
    )

    model = NADE(**vars(args)).to(args.device)
    # model = GRU(**vars(args)).to(args.device)
    # model = SimpleMADE(**vars(args)).to(args.device)
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info(f"TDVP for East model of {args.n} sites")
    logging.info(f"Hyperparameters: c={args.c:.2f}, logs={args.logs:.2f}, t={args.t}, dt={args.dt:.2f}")
    logging.info(f"Variational model: {model.__repr__()}")
    logging.info(f"Number of parameters: {num_params}")

    def loss_fn(log_probs):
        return log_probs.mean(0)

    def compute_loss(params, sample):
        batch = sample.unsqueeze(0)
        log_prob = functional_call(model, (params,), (batch,))
        loss = loss_fn(log_prob)
        return loss

    compute_grad = grad(compute_loss)
    compute_sample_grad = vmap(compute_grad, in_dims=(None, 0))

    logZts = 0.0
    step = 1
    model_ls = copy.deepcopy(model)  # model at last step

    if args.use_checkpoint:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint["step"]
        logZts = checkpoint["logZts"]
        # model_ls = copy.deepcopy(model)
        logging.info(f"Loading checkpoint from step {step}")

    logging.info("Start training")
    t_total = time.time()
    pbar = tqdm(range(step, 1 + int(args.t / args.dt)))
    for step in pbar:
        if step == 1:
            # use reinforce to initialize the model
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            loss_list = []
            t_per_step = time.time()
            for epoch in range(args.epochs * 10):
                optimizer.zero_grad()
                with torch.no_grad():
                    x = model.sample(args.batch_size)
                    tpx = transition_prob(x, model, args.c, args.s, args.dt, from_t0=(step == 1))
                log_prob = model(x)
                with torch.no_grad():
                    loss = log_prob - torch.log(tpx)
                loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
                loss_reinforce.backward()
                optimizer.step()
                loss_list.append(loss.mean().item())

            with torch.no_grad():
                logZ = -1 * np.mean(loss_list[-args.epochs // 20 :])  # use the last 5% of data
                logZts += logZ
                # model_ls = copy.deepcopy(model)
                pbar.set_description(f"step: {step}, logZ: {logZ:.4g}, logZts: {logZts:.4g}")
                with open(os.path.join(args.path, "output.txt"), "a", newline="\n") as f:
                    f.write(f"{step} {logZ} {logZts} {time.time() - t_per_step}\n")
        else:
            # use TDVP to update the model
            t_per_step = time.time()
            with torch.no_grad():
                x = model.sample(args.batch_size)
                ratio_factor = partial_t_log_prob(x, model, args.c, args.s, args.dt, from_t0=(step == 1))

            # calculate O_mat and F_vec
            params = {k: v.detach() for k, v in model.named_parameters()}
            grads = compute_sample_grad(params, x)  # d logP(x_i) / d theta_j, dict
            grads_flatten = torch.cat([torch.flatten(v, start_dim=1) for v in grads.values()], dim=1)  # N x M
            O_mat = grads_flatten / math.sqrt(args.batch_size)  # scaled by sqrt(N)
            F_vec = torch.einsum("nm,n->m", grads_flatten, ratio_factor) / args.batch_size
            updates_flatten = cholesky_solve(O_mat, F_vec, 1e-3)  # TODO: how to choose the optimal lambda?

            # convert updates_flatten into a dict
            updates, idx = {}, 0
            for k, v in model.named_parameters():
                updates[k] = updates_flatten[idx : idx + v.numel()].view(v.shape)
                idx += v.numel()

            # update parameters
            with torch.no_grad():
                for k, v in model.named_parameters():
                    v.data += args.dt * updates[k]

        with torch.no_grad():
            x = model.sample(args.batch_size)
            tpx = transition_prob(x, model, args.c, args.s, args.dt, from_t0=(step == 1))
            log_prob = model(x)
            loss = log_prob - torch.log(tpx)
            logZ = -1 * loss.mean().item()
            logZts += logZ
            # model_ls = copy.deepcopy(model)
            pbar.set_description(f"step: {step}, logZ: {logZ:.4g}, logZts: {logZts:.4g}")
            with open(os.path.join(args.path, "output.txt"), "a", newline="\n") as f:
                f.write(f"{step} {logZ} {logZts} {time.time() - t_per_step}\n")

        writer.add_scalar("logZ", logZ, step)
        writer.add_scalar("logZts", logZts, step)

        if args.save_model and step % int(1 / args.dt) == 0:
            t = int(step * args.dt)
            torch.save(
                {
                    "step": step,
                    "logZts": logZts,
                    "model_state_dict": model.state_dict(),
                },
                os.path.join(args.path, f"checkpoint_t{t}.pt"),
            )

    logging.info("Finish training")
    logging.info(f"Total time: {time.time() - t_total:.2f}s")


if __name__ == "__main__":
    main()
