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
from master_eq import transition_prob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import GRU, NADE, SimpleMADE


def main():
    # Bug exists in torch.set_default_dtype()
    # torch.set_default_dtype(torch.float64 if args.dtype=='double' else torch.float32)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and args.gpu > -1
    args.device = f"cuda:{args.gpu}" if use_cuda else "cpu"

    args.s = 10**args.logs
    if args.negative_s:
        args.s *= -1

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if args.use_tb:
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
    model = model.double() if args.dtype == "double" else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info(f"Evolving VAN for East model of {args.n} sites")
    logging.info(f"Hyperparameters: c={args.c:.2f}, logs={args.logs:.2f}, t={args.t}, dt={args.dt:.2f}")
    logging.info(f"Variational model: {model.__repr__()}")
    logging.info(f"Number of parameters: {num_params}")

    logZts = 0.0
    step = 1
    model_ls = copy.deepcopy(model)  # model at last step

    if args.use_checkpoint:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        logZts = checkpoint["logZts"]
        model_ls = copy.deepcopy(model)
        logging.info(f"Loading checkpoint from step {step}")

    logging.info("Start training")
    t_total = time.time()
    pbar = tqdm(range(step, 1 + int(args.t / args.dt)))
    for step in pbar:
        # for each step, we train for args.epochs * 10 if step == 1 else args.epochs
        loss_list = []
        t_per_step = time.time()
        for epoch in range(args.epochs * 10 if step == 1 else args.epochs):
            optimizer.zero_grad()
            with torch.no_grad():
                x = model.sample(args.batch_size)
                tpx = transition_prob(x, model_ls, args.c, args.s, args.dt, from_t0=(step == 1))
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
            model_ls = copy.deepcopy(model)
            pbar.set_description(f"step: {step}, logZ: {logZ:.4g}, logZts: {logZts:.4g}")
            with open(os.path.join(args.path, "output.txt"), "a", newline="\n") as f:
                f.write(f"{step} {logZ} {logZts} {time.time() - t_per_step}\n")

        if args.use_tb:
            writer.add_scalar("logZ", logZ, step)
            writer.add_scalar("logZts", logZts, step)

        if args.save_model and step % int(1 / args.dt) == 0:
            t = int(step * args.dt)
            torch.save(
                {
                    "step": step,
                    "logZts": logZts,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(args.path, f"checkpoint_t{t}.pt"),
            )

    if args.use_tb:
        writer.close()
    logging.info("Finish training")
    logging.info(f"Total time: {time.time() - t_total:.2f}s")


if __name__ == "__main__":
    main()
