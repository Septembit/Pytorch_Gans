import torch
import torchvision
import torchvision.transforms as transforms

def CIFAR(batch_size=32):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='/home/yachao-li/Downloads/CIFAR', train=True,
                                         transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
    return trainloader
