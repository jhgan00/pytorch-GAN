import os
import torch
import torchvision
from core.model import Generator, Discriminator
from collections import namedtuple

Nets = namedtuple("Nets", ["G", "D"])
Optimizers = namedtuple("Optimizers", ["G", "D"])
Schedulers = namedtuple("Schedulers", ["G", "D"])


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
            torch.optim.SGD(self.nets.G.parameters(), lr=args.lr, momentum=0.5),
            torch.optim.SGD(self.nets.D.parameters(), lr=args.lr, momentum=0.5)
        )

        for optim in self.optims:
            optim.param_groups[0]["step"] = 0

    def _update_lr(self):
        for optim in self.optims:
            lr = optim.param_groups[0]['lr']
            if lr > self.args.min_lr:
                new_lr = optim.param_groups[0]['lr'] * self.args.decay_factor
                optim.param_groups[0]['lr'] = new_lr if new_lr > self.args.min_lr else self.args.min_lr

    def _update_momentum(self, epoch):
        alpha = epoch / self.args.momentum_saturate
        if alpha < 0:
            alpha = 0.
        if alpha > 1:
            alpha = 1.
        for optim in self.optims:
            optim.param_groups[0]['momentum'] = self.args.momentum * (1 - alpha) + alpha * self.args.final_momentum

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

    def _sample_z(self, num_samples=None):
        num_samples = self.args.num_samples if num_samples is None else num_samples
        return torch.FloatTensor(num_samples, self.args.latent_dim) \
            .uniform_(-self.args.uniform_range, self.args.uniform_range) \
            .to(self.device)

    def _train_one_epoch(self, loader, epoch):

        for i, batch in enumerate(loader):

            x_real, _ = batch
            x_real = x_real.to(self.device)
            N, C, W, H = x_real.shape

            self._reset_grad()
            z = self._sample_z()
            x_fake = self.nets.G(z)
            real_out = self.nets.D(x_real)
            fake_out = self.nets.D(x_fake)

            d_loss_real = torch.nn.BCELoss()(real_out, torch.ones_like(real_out))
            d_loss_fake = torch.nn.BCELoss()(fake_out, torch.zeros_like(fake_out))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optims.D.step()

            self._reset_grad()
            z = self._sample_z()
            x_fake = self.nets.G(z)
            fake_out = self.nets.D(x_fake)
            g_loss = torch.nn.BCELoss()(fake_out, torch.ones_like(fake_out))
            g_loss.backward()
            self.optims.G.step()

            if i % 10 == 0:
                print(
                    f"epoch: {epoch + 1:04d}\t"
                    f"batch: {i:04d}\t"
                    f"G_loss: {g_loss.item():.4f}\t"
                    f"d_loss: {d_loss.item():.4f}\t"
                    f"lr: {self.optims.D.param_groups[0]['lr']:.6f}\t"
                    f"momentum: {self.optims.D.param_groups[0]['momentum']:.6f}\t"
                )

            self._update_lr()
        self._update_momentum(epoch)

    def train(self, loader):

        args = self.args

        if self.args.resume_epoch:
            self._restore_checkpoint(args.resume_epoch)

        for epoch in range(args.resume_epoch, args.epochs):
            self._train_one_epoch(loader, epoch)

            with torch.no_grad():

                z = self._sample_z()
                x_fake = self.nets.G(z)
                x_fake = x_fake.view(args.num_samples, -1, args.img_size, args.img_size)
                x_fake = torchvision.utils.make_grid(x_fake, nrow=int(args.num_samples ** 0.5))

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
        z = self._sample_z()
        x_fake = nets.G(z)
        x_fake = x_fake.view(args.num_samples, -1, args.img_size, args.img_size)
        x_fake = torchvision.utils.make_grid(x_fake, nrow=int(args.num_samples / 5))
        torchvision.utils.save_image(x_fake, os.path.join(save_path, "samples.png"))

        z1, z2 = self._sample_z(2)
        z = torch.empty(size=(args.num_samples, args.latent_dim), device=self.device)
        for i in range(args.num_samples):
            torch.lerp(z1, z2, i / args.num_samples, out=z[i])

        x_fake = nets.G(z)
        x_fake = x_fake.view(1, args.num_samples, -1, args.img_size, args.img_size)
        x_fake = torchvision.utils.make_grid(x_fake, nrow=1)
        torchvision.utils.save_image(x_fake, os.path.join(save_path, "latent.png"))
