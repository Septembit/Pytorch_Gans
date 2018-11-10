#Based on the paper ﻿Generative-Adversarial-Nets

import torch
from torch import nn
from dataloader import CIFAR
from torch.autograd import Variable
import torchvision
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)
        )
        self.fc = nn.Sequential(

            nn.Linear(256*4*4,1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024,1),
            nn.Sigmoid())
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(64,-1)

        x = self.fc(x)


        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


def main(epochs=100, bach_size=64 ):



    data_loader = CIFAR(batch_size=bach_size)
    criterion = nn.BCELoss().cuda()
    D_net = Discriminator().cuda()
    G_net = Generator().cuda()
    d_optimizer = torch.optim.RMSprop(D_net.parameters(), lr=0.0001)
    g_optimizer = torch.optim.RMSprop(G_net.parameters(), lr=0.0001)



    for epoch in range(epochs):
        ge_loss = []
        de_loss = []
        for i , data in enumerate(data_loader):

    # Training the discriminator



            real_img = Variable(data[0]).cuda()

            real_label = Variable(torch.ones(bach_size)).cuda()

            fake_label = Variable(torch.zeros(bach_size)).cuda()

            real_out = D_net(real_img)

            d_loss_real = criterion(real_out, real_label)

            z = Variable(torch.randn_like(real_img)).cuda()

            fake_img = G_net(z)

            fake_out = D_net(fake_img)

            d_loss_fake = criterion(fake_out, fake_label)


            d_loss = 0.5 * (d_loss_fake + d_loss_real)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
    # Training the generator
            z = Variable(torch.randn_like(real_img)).cuda()
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
                print(fake_img.size())
                #fake_img = fake_img.view([16, 3, 32, 32])
                torchvision.utils.save_image((fake_img), 'samples/' + str(i + 1) + '.jpg', normalize=True)
        de_loss = sum(de_loss)/len(de_loss)
        ge_loss = sum(ge_loss) / len(ge_loss)
        print("Loss at Epoch", epoch+1, "/", epochs,":G loss", ge_loss.item(), "D loss", de_loss.item() )




if __name__ == '__main__':
     main()



