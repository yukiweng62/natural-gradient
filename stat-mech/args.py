import argparse

parser = argparse.ArgumentParser()

# Physical config
parser.add_argument("--n", type=int, default=10, help="System size, default: 10")
parser.add_argument("--c", type=float, default=0.5, help="Equilibrium parameter, default: 0.5")
parser.add_argument("--logs", type=float, default=-1, help="log(tilted parameter), default: -1 (s=0.1)")
parser.add_argument("--negative-s", action="store_true", default=False, help="Use negative tilted parameter")
parser.add_argument("--t", type=int, default=100, help="Evolution time, default: 100")
parser.add_argument("--dt", type=float, default=0.1, help="delta t, default: 0.1")

# AutoRegressive model config
parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension of GRU/NADE, default: 64")

# Training config
parser.add_argument("--dtype", type=str, default="float", choices=["float", "double"], help="Data type, default: float")
parser.add_argument("--seed", type=int, default=1, help="Random seed, default: 1")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate, default: 1e-3")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs, default: 200")
parser.add_argument("--batch-size", type=int, default=1024, help="Batch size, default: 1024")
parser.add_argument("--gpu", type=int, default="0", help="GPU id, default: 0, -1 to disable")
parser.add_argument("--path", type=str, default="./out/test", help="Output path")
parser.add_argument("--lambd", type=float, default=1e-3, help="Damping factor, default: 1e-3")
parser.add_argument("--use-tb", action="store_true", help="Use tensorboard")

# Checkpoint config
parser.add_argument("--save-model", action="store_true", default=False, help="Save model")
parser.add_argument("--use-checkpoint", action="store_true", default=False, help="Use saved checkpoint")
parser.add_argument("--model-path", type=str, help="Checkpoint path")

args = parser.parse_args()
