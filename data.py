import random
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def preprocess_train(tensor):
    tensor -= tensor.mean().item()
    tensor = F.pad(tensor, pad=(4, 4, 4, 4), mode='constant', value=0)
    t = random.randrange(8)
    l = random.randrange(8)
    tensor = tensor[:, t:t+32, l:l+32]
    if random.random() < 0.5:
        tensor = transforms.functional.hflip(tensor)

    return tensor


def preprocess_test(tensor):
    tensor -= tensor.mean().item()
    return tensor


def get_dataloader(is_train, batch_size, path):
    if is_train:
        return DataLoader(
            datasets.CIFAR10(path,
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(preprocess_train)
                            ])),
            batch_size=batch_size,
            shuffle=True
        )
    else:
        return DataLoader(
            datasets.CIFAR10(path,
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(preprocess_test)
                            ])),
            batch_size=batch_size,
            shuffle=False
        )
