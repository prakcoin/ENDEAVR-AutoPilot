import torch
import torch.nn as nn
from .residual_block import ResidualBlock
from .separableconv import SeparableConv2d
from .convlstm import ConvLSTM

class AVModelLSTM(nn.Module):
    def __init__(self):
        super(AVModelLSTM, self).__init__()
        self.input_layer = nn.Conv2d(3, 8, kernel_size=5, padding=1, stride=4, padding_mode='reflect')

        self.norm = nn.LayerNorm(8)
        self.attention = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))
        self.act = nn.ReLU()

        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, stride=1, num_layers=2),
            ResidualBlock(in_channels=16, out_channels=16, kernel_size=3, stride=1, num_layers=2),
            nn.Dropout2d(0.2),
        )

        self.conv_lstm = ConvLSTM(input_dim=16, hidden_dim=16, kernel_size=(5, 5), num_layers=3, batch_first=True, bias=True, return_all_layers=False)

        self.dense_layers = nn.Sequential(
            nn.Linear(3753, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.output_layer = nn.Linear(10, 2)

    def use_attention(self, x):
        batch_size, channels, height, width = x.size()
        x_att = x.reshape(batch_size, channels, height * width).transpose(1, 2)
        x_att = self.norm(x_att)
        attention_output, _ = self.attention(x_att, x_att, x_att)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, channels, height, width)
        x = self.scale * attention_output + x
        return x

    def forward(self, img, hlc, speed, light):
        x = self.input_layer(img)
        x = self.use_attention(x)
        x = self.act(x)

        x = self.conv_layers(x)
        x = x.unsqueeze(1)

        _, last_states = self.conv_lstm(x)
        x =  last_states[0][0]

        #x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2) # GlobalAveragePooling2D
        x = x.reshape(x.size(0), -1)
        hlc = hlc.view(hlc.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        light = light.view(light.size(0), -1)
        x = torch.cat((x, hlc, speed, light), dim=1)

        x = self.dense_layers(x)
        x = self.output_layer(x)
        out = torch.sigmoid(x)
        return out