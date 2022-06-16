import torch
from torchsummary import summary
import torch
from torch import nn, optim
import argparse

class Generator(torch.nn.Module):
    def __init__(self, input_dim =512, out_dim=64, out_channels=1, noise_dim=200, kernel_size=(4,4,4) ,activation="sigmoid"):
        super(Generator, self).__init__()
        self.input_dim  = input_dim
        self.out_dim = out_dim
        self.in_dim = int(out_dim / 16)

        kernel_size = kernel_size
        conv1_out_channels = int(self.input_dim / int(2))
        conv2_out_channels = int(conv1_out_channels / int(2))
        conv3_out_channels = int(conv2_out_channels / int(2))

        self.linear = torch.nn.Linear(noise_dim, input_dim  * self.in_dim * self.in_dim * self.in_dim) #(1,200) -> (1,32769)
        # (1,32769)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels =input_dim , out_channels=conv1_out_channels, kernel_size=kernel_size,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels =conv1_out_channels, out_channels=conv2_out_channels, kernel_size=kernel_size,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels =conv2_out_channels, out_channels=conv3_out_channels, kernel_size=kernel_size,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels =conv3_out_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        if activation == "sigmoid":
            self.out = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Tanh()

    def project(self, x):
        """
        projects and reshapes latent vector to starting volume
        :param x: latent vector
        :return: starting volume
        """
        return x.view(-1, self.input_dim , self.in_dim, self.in_dim, self.in_dim) # 512*4*4*4

    def forward(self, x):
        x = self.linear(x)
        x = self.project(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.out(x)
device = torch.device("cuda") # PyTorch v0.4.0
model = Generator().to(device)
summary(model, (1,200))
print(model.conv1)