import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import TensorDataset, DataLoader
from typing import Callable, Optional, Tuple, Union, List
from tqdm import tqdm
from randaugment import RandAugmentMC
from randaugment_imagenet100 import RandAugmentMC as RandAugmentMC100
import os


class DualTransformDataset(Dataset):
    def __init__(self, dataset, transform_weak, transform_strong):
        self.dataset = dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.classes = dataset.classes
        self.targets = dataset.targets

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return [self.transform_weak(image), self.transform_strong(image)], label

    def __len__(self):
        return len(self.dataset)


def get_dual_dataset(dataset, data_path):
    if dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=None,
                                                      download=True)

        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        dual_dataset = DualTransformDataset(train_dataset, weak, strong)
        return dual_dataset

    elif dataset in ["imagenet100", "cub200"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                             transform=None)
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        dual_dataset = DualTransformDataset(train_dataset, weak, strong)
        return dual_dataset


def get_dataset(dataset, data_path):
    # assert dataset in ["cifar100", "imagenet100"]
    if dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            RandAugmentMC(n=1, m=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        cifar_transforms = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=strong,
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                     transform=cifar_transforms,
                                                     download=True)
    elif dataset in ["imagenet100", "cub200"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # data_path = os.path.join(data_path, "imagenet100")
        imagenet_tansforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomResizedCrop(224),
            RandAugmentMC(n=1, m=2),
            # RandAugmentMC100(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        # train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
        #                                      transform=imagenet_tansforms)
        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                             transform=strong)
        if dataset == "imagenet100":
            test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"),
                                                transform=imagenet_tansforms)
        elif dataset == "cub200":
            test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "test"),
                                                transform=imagenet_tansforms)

    return train_dataset, test_dataset


def get_dataset_std(dataset, data_path):
    # assert dataset in ["cifar100", "imagenet100"]
    if dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        cifar_transforms = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=cifar_transforms,
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                     transform=cifar_transforms,
                                                     download=True)
    elif dataset in ["imagenet100", "cub200"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # data_path = os.path.join(data_path, "imagenet100")
        imagenet_tansforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                             transform=imagenet_tansforms)
        if dataset == "imagenet100":
            test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"),
                                                transform=imagenet_tansforms)
        elif dataset == "cub200":
            test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "test"),
                                                transform=imagenet_tansforms)

    return train_dataset, test_dataset


def get_dataset_upbound(dataset, data_path):
    if dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transforms = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=train_transforms,
                                                      download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                     transform=test_transforms,
                                                     download=True)
    elif dataset in ["imagenet100", "cub200"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # data_path = os.path.join(data_path, "imagenet100")
        test_tansforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_tansforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                             transform=train_tansforms)
        if dataset == "imagenet100":
            test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"),
                                                transform=test_tansforms)
        elif dataset == "cub200":
            test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "test"),
                                                transform=test_tansforms)

    elif dataset in ["mini"]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]


        test_tansforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_tansforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),
                                             transform=train_tansforms)

        test_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"),
                                            transform=test_tansforms)

    return train_dataset, test_dataset


def split_dataset(dataset: Dataset, task_idx: List[int], tasks: list = None):
    assert len(dataset.classes) == sum([len(t) for t in tasks])
    current_task = torch.cat(tuple(tasks[i] for i in task_idx))
    mask = [(c in current_task) for c in dataset.targets]
    indexes = torch.tensor(mask).nonzero()
    task_dataset = Subset(dataset, indexes)
    return task_dataset


def split_dataset2(x, y, task_idx, tasks):
    current_task = torch.cat(tuple(tasks[i] for i in task_idx))
    mask = [(c in current_task) for c in y]
    x_task = x[mask]
    y_task = y[mask]
    return x_task, y_task


def get_pretrained_dataset(encoder, train_dataset, test_dataset, tau=1.0, return_means=False):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    encoder.eval()
    encoder.to(device)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8,
                             pin_memory=True)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    # encoder = nn.DataParallel(encoder)
    for x, y in tqdm(iter(train_loader), desc="pretrain on trainset"):
        x = x.to(device)
        z = encoder(x)
        x_train.append(z.cpu().detach().numpy())
        y_train.append(y.cpu().detach().numpy())
    for x, y in tqdm(iter(test_loader), desc="pretrain on testset"):
        x = x.to(device)
        z = encoder(x)
        x_test.append(z.cpu().detach().numpy())
        y_test.append(y.cpu().detach().numpy())

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    x_train = tau * x_train
    x_test = tau * x_test
    print("x_train.shape", x_train.shape, "y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape, "y_test.shape:", y_test.shape)
    # ds pretrained
    train_dataset_pretrained = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset_pretrained = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))

    if return_means:
        means = {}
        current_tasks = np.unique(y_train)
        for i in current_tasks:
            index_i = y_train == i
            x_train_i = x_train[index_i]
            mean_i = np.mean(x_train_i, axis=0)
            means.update({str(torch.tensor(i)): torch.tensor(mean_i)})
        return train_dataset_pretrained, test_dataset_pretrained, means
    else:
        return train_dataset_pretrained, test_dataset_pretrained


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset(dataset="imagenet100", data_path="/share/wenzhuoliu/torch_ds")
    print(train_dataset[0])
