import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(784, 512)
        self.norm1 = nn.LayerNorm(512)

        self.linear2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)

        self.linear3 = nn.Linear(256, 128)
        self.norm3 = nn.LayerNorm(128)

        self.linear4 = nn.Linear(128, 1)

        self.activation = nn.LeakyReLU(0.02)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.norm1(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.norm2(x)

        x = self.linear3(x)
        x = self.activation(x)
        x = self.norm3(x)

        x = self.linear4(x)

        x = self.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(100, 128)
        self.norm1 = nn. LayerNorm(128)

        self.linear2 = nn.Linear(128, 256)
        self.norm2 = nn.LayerNorm(256)

        self.linear3 = nn.Linear(256, 512)
        self.norm3 = nn.LayerNorm(512)

        self.linear4 = nn.Linear(512, 784)
        self.activation = nn.LeakyReLU(0.02)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.norm1(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.norm2(x)

        x = self.linear3(x)
        x = self.activation(x)
        x = self.norm3(x)

        x = self.linear4(x)

        x = self.tanh(x)

        return x
