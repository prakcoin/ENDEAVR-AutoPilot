import torch
import timm
import torch.nn as nn
from .residual_block import ResidualBlock
    
class RGBFeatureExtractor(nn.Module):
    def __init__(self, out_dim):
        super(RGBFeatureExtractor, self).__init__()
        self.input_layer = nn.Conv2d(3, out_dim // 8, kernel_size=5)
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=out_dim // 8, out_channels=out_dim // 8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim // 8, out_channels=out_dim // 4, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim // 4, out_channels=out_dim // 2, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim // 2, out_channels=out_dim, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=2, num_layers=2),
        )

    def forward(self, img):
        x = self.input_layer(img)
        out = self.conv_layers(x)
        return out

class DepthFeatureExtractor(nn.Module):
    def __init__(self, out_dim):
        super(DepthFeatureExtractor, self).__init__()
        self.input_layer = nn.Conv2d(1, out_dim // 8, kernel_size=5)
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=out_dim // 8, out_channels=out_dim // 8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim // 8, out_channels=out_dim // 4, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim // 4, out_channels=out_dim // 2, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim // 2, out_channels=out_dim, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=2, num_layers=2),
        )

    def forward(self, depth):
        x = self.input_layer(depth)
        out = self.conv_layers(x)
        return out


class RegNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("regnety_004", pretrained=True)
        self.backbone.fc = None
        self.backbone.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.backbone.head = nn.Sequential()

    def forward(self, x):
        return self.backbone(x)

class CNNTransformer(nn.Module):
    def __init__(self, out_dim=256, embed_dim=440, num_heads=4, depth=4):
        super(CNNTransformer, self).__init__()
        self.rgb_extractor = RegNetBackbone()

        self.pos_emb = nn.Parameter(torch.zeros(1, 8 * 16, embed_dim))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=(4 * embed_dim), activation='relu', norm_first=True, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=depth)
        self.ln = nn.LayerNorm(embed_dim)

        self.change_channel_conv_image = nn.Conv2d(embed_dim, out_dim, (1, 1))

        self.regression_head = nn.Sequential(
            nn.Linear(out_dim + 5, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, rgb, hlc, speed):
        rgb_features = self.rgb_extractor(rgb)

        rgb_bs, rgb_c, rgb_h, rgb_w = rgb_features.size()
        rgb_features_reshaped = rgb_features.reshape(rgb_bs, rgb_c, rgb_h * rgb_w).transpose(1, 2)

        transformer_features = rgb_features_reshaped + self.pos_emb
        transformer_output = self.transformer_encoder(transformer_features)
        transformer_output = self.ln(transformer_output)

        rgb_features_out = transformer_output[:, :rgb_h * rgb_w, :].transpose(1, 2).reshape(rgb_bs, rgb_c, rgb_h, rgb_w)
        rgb_features = rgb_features + rgb_features_out

        rgb_features = self.change_channel_conv_image(rgb_features)
        rgb_features = self.rgb_extractor.backbone.global_pool(rgb_features)
        rgb_features = torch.flatten(rgb_features, 1)

        speed = torch.flatten(speed, 1)
        hlc = torch.flatten(hlc, 1)
        x = torch.cat((rgb_features, speed, hlc), dim=1)
        x = self.regression_head(x)
        out = torch.sigmoid(x)
        return out