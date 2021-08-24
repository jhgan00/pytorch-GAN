import os
import glob
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):

    def __init__(self, root, dataset, transform=None):
        super(CustomDataset, self).__init__()
        self.root = root
        self.dataset = dataset
        self.image_paths = glob.glob(os.path.join(root, dataset, "*"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem(self, idx):
        x = read_image(self.image_paths[idx])
        if self.transform:
            x = self.transform(x)
        return x


def get_train_loader(root, dataset, img_size, batch_size, num_workers):

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset in {"MNIST", "CIFAR10"}:
        train_dataset = getattr(datasets, dataset)(root, download=True, train=True, transform=transform)

    else:
        train_dataset = None
        pass

    print(f"total {len(train_dataset)} examples ...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader
