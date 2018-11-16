#Based on the paper Conditional Generative Adversarial Nets
import torch
import torchvision
import numpy as np
import visdom
from torch import nn
from torch.autograd import Variable

import argparse
from dataloader import MINIST
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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

    # visualization By Visdom
    vis = visdom.Visdom()
    lineg = vis.line(Y=np.arange(10), env="G_Loss")
    lined = vis.line(Y=np.arange(10), env="D_Loss")

    data_loader = MINIST(batch_size=opt.batchsize,resize=False)
    criterion = nn.BCELoss().to(device)
    D_net = discriminator( ngpu=1, n_class=10).to(device)
    G_net = generator(ngc=opt.ngf, ngpu=1, nz=opt.nz, nc=10).to(device)
    G_net.apply(weights_init)
    D_net.apply(weights_init)
    d_optimizer = torch.optim.Adam(D_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    g_optimizer = torch.optim.Adam(G_net.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    epochs = opt.epoch
    G_loss = []
    D_loss = []
    for epoch in range(epochs):
        ge_loss = []
        de_loss = []
        for i, data in enumerate(data_loader):

            # Training the discriminator

            d_optimizer.zero_grad()
            real_img = Variable(data[0]).to(device)
            label = data[1].numpy()
            label = (np.arange(10) == label[:,None]).astype(np.float32)
            label = Variable(torch.from_numpy()).to(device)

              #to be continued


            real_label = Variable(torch.ones(opt.batchsize)).to(device)

            fake_label = Variable(torch.zeros(opt.batchsize)).to(device)

            real_out = D_net(real_img)

            d_loss_real = criterion(real_out, real_label)

            z = Variable(torch.randn(opt.batchsize, opt.nz)).to(device)

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
            # print("Loss at iteration", i + 1, "/", len(data_loader), ":G loss", g_loss.item(), "D loss", d_loss.item())

            # save fake iamges and visulization data
            if i % 1000 == 0:
                print(fake_img.size())
                torchvision.utils.save_image((fake_img), 'samples/' + str(i + 1) + '.jpg', normalize=True)
                torchvision.utils.save_image((real_img), 'samples/' + str(i + 1) + 'gt.jpg', normalize=True)
        de_loss = sum(de_loss) / len(de_loss)
        ge_loss = sum(ge_loss) / len(ge_loss)

        D_loss.append(de_loss.item())
        G_loss.append(ge_loss.item())
        vis.line(Y=np.array(G_loss), X=np.arange(len(G_loss)),
                 opts=dict(title="G_loss"), update="new", win=lineg,
                 env="G_Loss")

        vis.line(Y=np.array(D_loss), X=np.arange(len(D_loss)),
                 opts=dict(title="D_loss"), update="new", win=lined,
                 env="D_Loss")
        print("Loss at Epoch", epoch + 1, "/", epochs, ":G loss", ge_loss.item(), "D loss", de_loss.item())


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

class generator(nn.Module):
    def __init__(self, ngpu, nc=10, nz=100, ngc=64):
        super(generator, self).__init__()
        self.ngpu = ngpu

        self.gen = nn.Sequential(

            nn.Linear(nz+nc, ngc),
            nn.BatchNorm1d(ngc),
            nn.ReLU(True),

            nn.Linear(ngc, ngc*2),
            nn.BatchNorm1d(ngc*2),
            nn.ReLU(True),

            nn.Linear(ngc*2, ngc*4),
            nn.BatchNorm1d(ngc*4),
            nn.ReLU(True),

            nn.Linear(ngc*4, 784),
            nn.Tanh() )

    def forward(self, x):

        if self.ngpu > 1:
            x = nn.parallel.data_parallel(self.gen, x, range(self.ngpu))
        else:
            x = self.dis(x)

        return x.view(-1, 1, 28, 28)


if __name__ == '__main__':
     main()




