import torch
import torch.nn as nn
from .residual_block import ResidualBlock

class AVModel(nn.Module):
    def __init__(self):
        super(AVModel, self).__init__()
        # self.norm = nn.LayerNorm(64)
        # self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.5, batch_first=True)
        # self.scale = nn.Parameter(torch.zeros(1))

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, bias=False),
            nn.SELU(),
        )

        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, num_layers=2, pool=True, short=True),
            nn.Dropout2d(0.2),
            ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, num_layers=2, pool=True, short=True),
            nn.Dropout2d(0.2),
            ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, num_layers=2, pool=True, short=True),
            nn.Dropout2d(0.2),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(256, 50, bias=False),
            nn.SELU(),
            nn.Linear(50, 10, bias=False),
            nn.SELU(),
            nn.Dropout(0.5),
        )

        self.output_layer = nn.Linear(10, 2)

    def forward(self, x):
        # batch_size, channels, height, width = x.size()
        # x_att = x.reshape(batch_size, channels, height * width).transpose(1, 2)
        # x_att = self.norm(x_att)
        # attention_output, _ = self.attention(x_att, x_att, x_att)
        # attention_output = attention_output.transpose(1, 2).reshape(batch_size, channels, height, width)
        # x = self.scale * attention_output + x

        x = self.input_layer(x)
        x = self.conv_layers(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2) # GlobalAveragePooling2D
        x = self.dense_layers(x)
        x = self.output_layer(x)
        out = torch.sigmoid(x)
        return out