import torch

from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.dis = nn.Sequential(

            nn.Linear(3072, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.dis(x)

        return x

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator,self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 3702),
            nn.Tanh()
            )

    def forward(self, x):
        x = self.gen(x)

        return x

criterion = nn.BCELoss()
d_optimizer = torch.optim.RMSprop(Discriminator.parameters(), lr=0.0001)
g_optimizer = torch.optim.RMSprop(Generator.parameters(), lr=0.0001)

#Training the discriminator
