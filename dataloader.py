import torch
import torchvision
import torchvision.transforms as transforms

def CIFAR(batch_size=32,imagesize=32):

    transform = transforms.Compose(
        [ transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='~/Downloads/CIFAR', train=True,download=True,
                                         transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4,drop_last=True)
    return trainloader

def MINIST(batch_size=32, resize=True):
    if resize:
        transform = transforms.Compose(
            [ transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    trainset = torchvision.datasets.MNIST(root='~/Downloads/MINIST', train=True, download=True,
                                         transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4,drop_last=True)
    return trainloader

