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

    parser.add_argument("--epochs", type=int, default=250,
                        help="Total number of epochs")
    parser.add_argument("--resume_epoch", type=int, default=0,
                        help="Resume training from this point")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save checkpoint every n epochs")

    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimensions of latent space")
    parser.add_argument("--uniform_range", type=float, default=3 ** 0.5,
                        help="Range of latent vector elements: Uniform(-x, x)")
    parser.add_argument("--img_size", type=int, default=28,
                        help="Size of input images")
    parser.add_argument("--num_channels", type=int, default=1,
                        help="Number of channels in input image")

    parser.add_argument("--lr", type=float, default=1e-1,
                        help="Initial value for the learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="The learning rate will be clipped to be at least this value")
    parser.add_argument("--decay_factor", type=float, default=1 / (1+4e-6),
                        help="Multiplicative factor of learning rate decay")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="Initial value for the momentum coefficient")
    parser.add_argument("--final_momentum", type=float, default=.7,
                        help="The momentum coefficient to use at the end of learning")
    parser.add_argument("--momentum_saturate", type=int, default=250,
                        help="The epoch on which the moment should reach its final value")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Size of minibatch")

    parser.add_argument("--root", type=str, default="./data",
                        help="Root directory where datasets will be stored")
    parser.add_argument("--dataset", type=str, choices=['MNIST', 'CIFAR10', 'custom'], default="MNIST",
                        help="Dataset to be used for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader processes")

    parser.add_argument("--num_samples", type=int, default=25,
                        help="Number of fake images to generate")
    parser.add_argument("--test_checkpoint", type=int, default=250,
                        help="Checkpoint number to use for testing")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Path to save model checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples",
                        help="Path to save generated images")
    parser.add_argument("--test_dir", type=str, default="test",
                        help="Path to save test results")

    parser.add_argument("--seed", type=int, default=777,
                        help="Random seed")

    args = parser.parse_args()

    main(args)
