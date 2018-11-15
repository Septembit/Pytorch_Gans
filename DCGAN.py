#Based on the paper Deep Convolution ﻿Generative-Adversarial-Nets

import torch
from torch import nn
from dataloader import CIFAR, MINIST
from torch.autograd import Variable
import torchvision
import argparse
import visdom
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, ngpu,ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:

            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class Generator(nn.Module):
    def __init__(self, ngpu,ngf,nz, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,   nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument("--data", type=str, default="MINIST", help="dataset used to train" )
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')


    opt = parser.parse_args()
    print(opt)
    #visualization By Visdom
    vis = visdom.Visdom()
    lineg = vis.line(Y=np.arange(10), env="G_Loss")
    lined = vis.line(Y=np.arange(10), env="D_Loss")



    #load data and initialize the networks and optimizers
    if opt.data == "CIFAR":
        data_loader = CIFAR(batch_size=opt.batchsize)
        nc = 3
    else:
        data_loader = MINIST(batch_size=opt.batchsize)
        nc = 1
    criterion = nn.BCELoss().to(device)
    D_net = Discriminator(ndf=opt.ndf,ngpu=1, nc=nc).to(device)
    G_net = Generator(ngf=opt.ngf,ngpu=1,nz=opt.nz, nc=nc).to(device)
    G_net.apply(weights_init)
    D_net.apply(weights_init)
    d_optimizer = torch.optim.Adam(D_net.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))
    g_optimizer = torch.optim.Adam(G_net.parameters(), lr=opt.lr, betas=(0.9,0.999))

    epochs=opt.epoch
    G_loss = []
    D_loss = []
    for epoch in range(epochs):
        ge_loss = []
        de_loss = []
        for i , data in enumerate(data_loader):

    # Training the discriminator

            d_optimizer.zero_grad()
            real_img = Variable(data[0]).to(device)

            real_label = Variable(torch.ones(opt.batchsize)).to(device)

            fake_label = Variable(torch.zeros(opt.batchsize)).to(device)

            real_out = D_net(real_img)

            d_loss_real = criterion(real_out, real_label)

            z = Variable(torch.randn(opt.batchsize, opt.nz, 1, 1,)).to(device)

            fake_img = G_net(z)

            fake_out = D_net(fake_img)

            d_loss_fake = criterion(fake_out, fake_label)


            d_loss = 0.5 * (d_loss_fake + d_loss_real)

            d_loss.backward()
            d_optimizer.step()
    # Training the generator
            g_optimizer.zero_grad()

            z = Variable(torch.randn(opt.batchsize, opt.nz, 1, 1, )).to(device)
            fake_img = G_net(z)
            output = D_net(fake_img)
            g_loss = criterion(output, real_label)

            g_loss.backward()
            g_optimizer.step()

            de_loss.append(d_loss)
            ge_loss.append(g_loss)
            #print("Loss at iteration", i + 1, "/", len(data_loader), ":G loss", g_loss.item(), "D loss", d_loss.item())

            #save fake iamges and visulization data
            if i % 1000 == 0:
                print(fake_img.size())
                torchvision.utils.save_image((fake_img), 'samples/' + str(i + 1) + '.jpg', normalize=True)
                torchvision.utils.save_image((real_img), 'samples/' + str(i + 1) + 'gt.jpg', normalize=True)
        de_loss = sum(de_loss)/len(de_loss)
        ge_loss = sum(ge_loss) / len(ge_loss)


        D_loss.append(de_loss.item())
        G_loss.append(ge_loss.item())
        vis.line(Y= np.array(G_loss), X= np.arange(len(G_loss)),
                  opts=dict( title="G_loss"),update="new", win=lineg,
                 env="G_Loss")

        vis.line(Y=np.array(D_loss),X= np.arange(len(D_loss)),
                  opts=dict(title="D_loss"),update="new", win=lined,
                 env="D_Loss")
        print("Loss at Epoch", epoch+1, "/", epochs,":G loss", ge_loss.item(), "D loss", de_loss.item() )




if __name__ == '__main__':
     main()




