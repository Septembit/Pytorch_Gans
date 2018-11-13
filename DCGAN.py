#Based on the paper ﻿Generative-Adversarial-Nets

import torch
from torch import nn
from dataloader import CIFAR
from torch.autograd import Variable
import torchvision
import argparse

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self, ngpu,ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
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
    def __init__(self, ngpu,ngf,nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
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
            nn.ConvTranspose2d(    ngf,   3, 4, 2, 1, bias=False),
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
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')


    opt = parser.parse_args()
    print(opt)

    data_loader = CIFAR(batch_size=opt.bachsize)
    criterion = nn.BCELoss().cuda()
    D_net = Discriminator(ndf=opt.ndf,ngpu=1).cuda()
    G_net = Generator(ngf=opt.ngf,ngpu=1,nz=opt.nz).cuda()
    G_net.apply(weights_init)
    D_net.apply(weights_init)
    d_optimizer = torch.optim.Adam(D_net.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))
    g_optimizer = torch.optim.Adam(G_net.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))

    epochs=opt.epoch
    for epoch in range(epochs):
        ge_loss = []
        de_loss = []
        for i , data in enumerate(data_loader):

    # Training the discriminator

            d_optimizer.zero_grad()
            real_img = Variable(data[0]).cuda()

            real_label = Variable(torch.ones(opt.bachsize)).cuda()

            fake_label = Variable(torch.zeros(opt.bachsize)).cuda()

            real_out = D_net(real_img)

            d_loss_real = criterion(real_out, real_label)

            z = Variable(torch.randn(opt.batchSize, opt.nz, 1, 1,)).cuda()

            fake_img = G_net(z)

            fake_out = D_net(fake_img)

            d_loss_fake = criterion(fake_out, fake_label)


            d_loss = 0.5 * (d_loss_fake + d_loss_real)

            d_loss.backward()
            d_optimizer.step()
    # Training the generator
            g_optimizer.zero_grad()

            z = Variable(torch.randn(opt.batchSize, opt.nz, 1, 1, )).cuda()
            fake_img = G_net(z)
            output = D_net(fake_img)
            g_loss = criterion(output, real_label)

            g_loss.backward()
            g_optimizer.step()

            de_loss.append(d_loss)
            ge_loss.append(g_loss)
            #print("Loss at iteration", i + 1, "/", len(data_loader), ":G loss", g_loss.item(), "D loss", d_loss.item())
            if i % 1000 == 0:
                print(fake_img.size())
                #fake_img = fake_img.view([16, 3, 32, 32])
                torchvision.utils.save_image((fake_img), 'samples/' + str(i + 1) + '.jpg', normalize=True)
        de_loss = sum(de_loss)/len(de_loss)
        ge_loss = sum(ge_loss) / len(ge_loss)
        print("Loss at Epoch", epoch+1, "/", epochs,":G loss", ge_loss.item(), "D loss", de_loss.item() )




if __name__ == '__main__':
     main()




