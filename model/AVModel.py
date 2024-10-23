import torch
import torch.nn as nn
from .residual_block import ResidualBlock
from .separableconv import SeparableConv2d
from .convlstm import ConvLSTM

class AVModelLSTM(nn.Module):
    def __init__(self):
        super(AVModelLSTM, self).__init__()
        self.input_layer = nn.Conv2d(6, 8, kernel_size=5, padding=1, stride=4, padding_mode='reflect')

        self.norm = nn.LayerNorm(8)
        self.attention = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))
        self.act = nn.ReLU()

        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, num_layers=2),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, num_layers=2),
            nn.Dropout2d(0.2),
        )

        self.conv_lstm = ConvLSTM(input_dim=32, hidden_dim=32, kernel_size=(5, 5), num_layers=3, batch_first=True, bias=True, return_all_layers=False)

        self.dense_layers = nn.Sequential(
            nn.Linear(7497, 50),
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
    
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.input_layer = nn.Conv2d(3, 8, kernel_size=5, padding=1, stride=2)

        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, num_layers=2),
            ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, num_layers=2),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def forward(self, img):
        x = self.input_layer(img)
        x = self.conv_layers(x)
        out = x.reshape(x.size(0), -1)
        return out


class CNNTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=512, num_heads=8, depth=6, mlp_ratio=4.0):
        super(CNNTransformer, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor()

        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim))
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(521),
            nn.Linear(521, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 2)
        )

    def forward(self, img, hlc, light, speed):
        img_features = self.cnn_extractor(img)

        img_features = img_features.unsqueeze(0)
        img_features = self.transformer(img_features) 
        img_features = img_features.squeeze(0)

        hlc = hlc.view(hlc.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        light = light.view(light.size(0), -1)

        x = torch.cat((img_features, hlc, speed, light), dim=1)
        x = self.regression_head(x)
        out = torch.sigmoid(x)
        return out