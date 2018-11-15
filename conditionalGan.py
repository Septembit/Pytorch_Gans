#Based on the paper Conditional Generative Adversarial Nets
import torch
import torchvision
import numpy as np
import visdom
from torch import nn
import argparse

def main():

    parser = argparse.ArgumentParser(description="Based on the paper Conditional Generative Adversarial Nets")

    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument("--n_class", type=int, default=10, help="the number of class")
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    opt = parser.parse_args()
    print(opt)


class discriminator(nn.Module):

    def __init__(self, n_class,ngpu, dorp_rate=None):
        super(discriminator, self).__init__()
        self.ngpu = ngpu
        self.dis = nn.Sequential(

            nn.Linear(784 + n_class,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
            )
        if dorp_rate != None:
            self.dis.add_module("drop_out", nn.Dropout(dorp_rate))

    def forward(self, input):

        if self.ngpu > 1:
            x = nn.parallel.data_parallel(self.dis,input, range(self.ngpu) )
        else:
            x = self.dis(input)

        return x



if __name__ == '__main__':
     main()




