from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_train_loader(root, dataset, img_size, batch_size, num_workers):

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
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
