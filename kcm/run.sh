# exact value
python exact_1d.py

# evolving VAN, using REINFORCE
python train_1d_reinforce.py --dtype double --n 10 --logs -1 --path './out/reinforce_nade_n10_seed1_h64_logs-1' --use-tb

# evolving VAN, using natural gradient
python train_1d_natural_grad.py --dtype double --n 10 --logs -1 --epochs 20 --lr 1 --path './out/natural_grad_nade_n10_seed1_h64_logs-1' --use-tb