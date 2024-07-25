import torch
import torch.nn as nn
from .residual_block import ResidualBlock
from .separableconv import SeparableConv2d
from .convlstm import ConvLSTM

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
    
class AVModelLSTM(nn.Module):
    def __init__(self):
        super(AVModelLSTM, self).__init__()
        self.conv_layers = nn.Sequential(
            SeparableConv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, bias=False),
            nn.SELU(),
            SeparableConv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, bias=False),
            nn.SELU(),
            SeparableConv2d(in_channels=16, out_channels=24, kernel_size=5, stride=2, bias=False),
            nn.SELU(),
            SeparableConv2d(in_channels=24, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.SELU(),
            SeparableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False),
            nn.SELU(),
            SeparableConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False),
            nn.SELU(),
            nn.Dropout2d(0.2),
        )

        self.conv_lstm = ConvLSTM(input_dim=32, hidden_dim=32, kernel_size=(5, 5), num_layers=3, batch_first=True, bias=True, return_all_layers=False)

        self.dense_layers = nn.Sequential(
            nn.Linear(2693, 50, bias=False),
            nn.SELU(),
            nn.Linear(50, 10, bias=False),
            nn.SELU(),
            nn.Dropout(0.5),
        )

        self.output_layer = nn.Linear(10, 2)

    def forward(self, img, hlc, speed):
        x = self.conv_layers(img)
        x = x.unsqueeze(1)

        _, last_states = self.conv_lstm(x)
        x =  last_states[0][0]

        #x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2) # GlobalAveragePooling2D
        x = x.reshape(x.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        hlc = hlc.view(hlc.size(0), -1)
        x = torch.cat((x, speed, hlc), dim=1)

        x = self.dense_layers(x)
        x = self.output_layer(x)
        out = torch.sigmoid(x)
        return out