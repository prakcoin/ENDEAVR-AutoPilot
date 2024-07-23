import torch
import torch.nn as nn
from .residual_block import ResidualBlock

class AVModel(nn.Module):
    def __init__(self):
        super(AVModel, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, bias=False),
            nn.SELU(),
        )

        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, num_layers=2, pool=True, short=True),
            ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, num_layers=2, pool=True, short=True),
            ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, num_layers=4, pool=True, short=True),
            ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, num_layers=4, pool=True, short=True),
            nn.Dropout2d(0.2),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(258, 50, bias=False),
            nn.SELU(),
            nn.Linear(50, 10, bias=False),
            nn.SELU(),
            nn.Dropout(0.5),
        )

        self.output_layer = nn.Linear(10, 2)

    def forward(self, img, hlc, speed):
        x = self.input_layer(img)
        x = self.conv_layers(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2) # GlobalAveragePooling2D

        speed = speed.view(speed.size(0), -1)
        hlc = hlc.view(hlc.size(0), -1)
        x = torch.cat((x, speed, hlc), dim=1)

        x = self.dense_layers(x)
        x = self.output_layer(x)
        out = torch.sigmoid(x)
        return out