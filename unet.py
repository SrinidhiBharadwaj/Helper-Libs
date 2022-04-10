import numpy as np
import torch
from torch.utils.data import dataloader
import torch.nn as nn
import torchvision

class block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class encoder(nn.Module):
    def __init__(self, c_list):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.blocks = nn.ModuleList([block(c_list[i], c_list[i+1]) for i in range(len(c_list)-1)])

    def forward(self, x):
        filters = []
        for block in self.blocks:
            x = block(x)
            filters.append(x)
            x = self.pool(x)
        return filters


class decoder(nn.Module):
    def __init__(self, c_list):
        super().__init__()
        self.up_cov = nn.ModuleList([nn.ConvTranspose2d(c_list[i], c_list[i+1], kernel_size=2, stride=2)
                                                                    for i in range(len(c_list)-1)])
        self.conv_blocks = nn.ModuleList([block(c_list[i], c_list[i+1]) for i in range(len(c_list)-1)])
        

    def forward(self, x, features):    
        for i in range(len(self.conv_blocks)):
            x = self.up_cov[i](x)
            x = torch.cat([x, torchvision.transforms.CenterCrop([x.shape[2], x.shape[3]])(features[i])], dim=1)
            x = self.conv_blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self, c_list, num_classes):
        super().__init__()
        self.enc_blocks = encoder(c_list)
        self.dec_blocks = decoder(c_list[::-1][:-1])
        self.last_layer = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        filters = self.enc_blocks(x)
        x = self.dec_blocks(filters[-1], filters[::-1][1:])
        return self.last_layer(x)


if __name__ == "__main__":

    image = torch.randn(1, 3, 572, 572)
    channel_list = [3, 64, 128, 256, 512, 1024]
    enc_block = encoder(channel_list)
    filters = enc_block(image)
    
    x = filters[-1]
    dec_block = decoder(channel_list[::-1][:-1])
    x = dec_block(x, filters[::-1][1:])
    print(x.size())

    unet = UNet(channel_list, 1)
    x = torch.randn(1, 3, 572, 572)
    print(unet(x).shape)