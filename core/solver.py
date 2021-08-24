import os
import torch
import torchvision
from core.model import Generator, Discriminator
from collections import namedtuple

Nets = namedtuple("Nets", ["G", "D"])
Optimizers = namedtuple("Optimizers", ["G", "D"])


class Solver(object):

    def __init__(self, args):
        super(Solver, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets = Nets(
            Generator(args.img_size, args.num_channels, args.latent_dim).to(self.device),
            Discriminator(args.img_size, args.num_channels).to(self.device)
        )
        self.optims = Optimizers(
            torch.optim.Adam(self.nets.G.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2)),
            torch.optim.Adam(self.nets.D.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        )

    def _save_checkpoint(self, checkpoint_num):
        save_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"epoch_{checkpoint_num :04d}.pt")
        print(f"[*] Saving checkpoint at {save_path} ...", end=" ")
        torch.save({
            'G': self.nets.G.state_dict(),
            'D': self.nets.D.state_dict(),
            'optimizerG': self.optims.G.state_dict(),
            'optimizerD': self.optims.D.state_dict(),
        }, save_path)
        print("done")

    def _restore_checkpoint(self, checkpoint_num):

        checkpoint_path = os.path.join(self.args.checkpoint_dir, self.args.dataset, f"epoch_{checkpoint_num:04d}.pt")
        print(f"[*] Restoring checkpoint at {checkpoint_path} ...", end=" ")
        checkpoint = torch.load(checkpoint_path)
        self.nets.G.load_state_dict(checkpoint['G'])
        self.nets.D.load_state_dict(checkpoint['D'])
        self.optims.G.load_state_dict(checkpoint['optimizerG'])
        self.optims.D.load_state_dict(checkpoint['optimizerD'])
        print("done")

    def _reset_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def _train_one_epoch(self, nets, loader, optims, epoch, args):

        for i, batch in enumerate(loader):

            x_real, _ = batch
            x_real = x_real.to(self.device)
            N, C, W, H = x_real.shape

            self._reset_grad()
            z = torch.rand(N, args.latent_dim, device=self.device)
            x_fake = nets.G(z)
            real_out = nets.D(x_real)
            fake_out = nets.D(x_fake)
            d_loss = -torch.mean(torch.log(real_out) + torch.log(1 - fake_out))
            d_loss.backward()
            optims.D.step()

            self._reset_grad()
            z = torch.rand(N, args.latent_dim, device=self.device)
            x_fake = nets.G(z)
            fake_out = nets.D(x_fake)
            g_loss = -torch.mean(torch.log(fake_out))
            g_loss.backward()
            optims.G.step()

            if i % 10 == 0:
                print(f"epoch: {epoch+1:04d}\tbatch: {i:04d}\tG_loss: {g_loss.item():.4f}\td_loss: {d_loss.item():.4f}")

    def train(self, loader):

        args = self.args
        nets = self.nets
        optims = self.optims

        if self.args.resume_epoch:
            self._restore_checkpoint(args.resume_epoch)

        for epoch in range(args.resume_epoch, args.epochs):
            self._train_one_epoch(nets, loader, optims, epoch, args)

            with torch.no_grad():

                z = torch.rand(args.n_samples, self.args.latent_dim, device=self.device)
                x_fake = self.nets.G(z)
                x_fake = x_fake.view(args.n_samples, -1, args.img_size, args.img_size)
                x_fake = torchvision.utils.make_grid(x_fake, nrow=int(args.n_samples ** 0.5))

                save_dir = os.path.join(args.sample_dir, args.dataset)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"epoch_{epoch+1:04d}.png")
                torchvision.utils.save_image(x_fake, save_path)

            if (epoch + 1) % self.args.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1)

    @torch.no_grad()
    def test(self):

        args = self.args
        nets = self.nets

        if not os.path.exists(args.test_dir):
            os.mkdir(args.test_dir)

        self._restore_checkpoint(args.test_checkpoint)
        save_path = os.path.join(args.test_dir, args.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        z = torch.rand(args.n_samples, args.latent_dim, device=self.device)
        x_fake = nets.G(z)
        x_fake = x_fake.view(25, -1, args.img_size, args.img_size)
        x_fake = torchvision.utils.make_grid(x_fake, nrow=5)
        torchvision.utils.save_image(x_fake, os.path.join(save_path, "samples.png"))

        z1, z2 = torch.rand(2, 1, args.latent_dim, device=self.device)
        z = torch.empty(size=(args.n_samples, args.latent_dim), device=self.device)
        for i in range(args.n_samples):
            torch.lerp(z1, z2, i / args.n_samples, out=z[i])

        x_fake = nets.G(z)
        x_fake = x_fake.view(args.n_samples, -1, args.img_size, args.img_size)
        x_fake = torchvision.utils.make_grid(x_fake, nrow=1)
        torchvision.utils.save_image(x_fake, os.path.join(save_path, "latent.png"))
