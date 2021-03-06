#Based on the paper ﻿Generative-Adversarial-Nets

import torch
from torch import nn
from dataloader import CIFAR
from torch.autograd import Variable
import torchvision
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(

            nn.Linear(3072, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.dis(x)

        return x

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3072),
            nn.Tanh()
            )

    def forward(self, x):
        x = self.gen(x)

        return x


def main(epochs=100, bach_size=64, z_dim=128 ):



    data_loader = CIFAR(batch_size=bach_size)
    criterion = nn.BCELoss().cuda()
    D_net = Discriminator().cuda()
    G_net = Generator(z_dim).cuda()
    d_optimizer = torch.optim.RMSprop(D_net.parameters(), lr=0.0001)
    g_optimizer = torch.optim.RMSprop(G_net.parameters(), lr=0.0001)



    for epoch in range(epochs):
        ge_loss = []
        de_loss = []
        for i , data in enumerate(data_loader):

    # Training the discriminator

            data[0] = data[0].view(-1,3072)

            real_img = Variable(data[0]).cuda()

            real_label = Variable(torch.ones(bach_size)).cuda()

            fake_label = Variable(torch.zeros(bach_size)).cuda()

            real_out = D_net(real_img)

            d_loss_real = criterion(real_out, real_label)

            z = Variable(torch.randn(bach_size, z_dim)).cuda()

            fake_img = G_net(z)

            fake_out = D_net(fake_img)

            d_loss_fake = criterion(fake_out, fake_label)


            d_loss = 0.5 * (d_loss_fake + d_loss_real)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
    # Training the generator
            z = Variable(torch.randn(bach_size, z_dim)).cuda()
            fake_img = G_net(z)
            output = D_net(fake_img)
            g_loss = criterion(output, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            de_loss.append(d_loss)
            ge_loss.append(g_loss)
            #print("Loss at iteration", i + 1, "/", len(data_loader), ":G loss", g_loss.item(), "D loss", d_loss.item())
            if i % 1000 == 0:
                #print(fake_img.size())
                fake_img = fake_img.view([bach_size, 3, 32, 32])
                torchvision.utils.save_image((fake_img), 'samples/' + str(i + 1) + '.jpg', normalize=True)
        de_loss = sum(de_loss)/len(de_loss)
        ge_loss = sum(ge_loss) / len(ge_loss)
        print("Loss at Epoch", epoch+1, "/", epochs,":G loss", ge_loss.item(), "D loss", de_loss.item() )




if __name__ == '__main__':
     main()




