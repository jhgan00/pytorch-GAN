import argparse
import torch
import numpy as np
import random
from core.data_loader import get_train_loader
from core.solver import Solver


def main(args):

    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    solver = Solver(args)

    if args.mode == "train":
        print("loading dataset ...", end=" ")
        train_loader = get_train_loader(
            root=args.root,
            dataset=args.dataset,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print("done")
        solver.train(train_loader)

    if args.mode == "test":
        solver.test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=['train', 'test'], default="train")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=50)

    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--num_channels", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta_1", type=float, default=0.5)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--dataset", type=str, choices=['MNIST', 'CIFAR10', 'custom'], default="MNIST")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--n_samples", type=int, default=25)
    parser.add_argument("--test_checkpoint", type=int, default=25)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--test_dir", type=str, default="test")

    parser.add_argument("--seed", type=int, default=777)

    args = parser.parse_args()

    main(args)
