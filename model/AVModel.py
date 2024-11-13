import torch
import torch.nn as nn
from .residual_block import ResidualBlock
    
class RGBFeatureExtractor(nn.Module):
    def __init__(self):
        super(RGBFeatureExtractor, self).__init__()
        self.input_layer = nn.Conv2d(3, 8, kernel_size=5)
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, num_layers=2),
        )

    def forward(self, img):
        x = self.input_layer(img)
        out = self.conv_layers(x)
        return out

class DepthFeatureExtractor(nn.Module):
    def __init__(self):
        super(DepthFeatureExtractor, self).__init__()
        self.input_layer = nn.Conv2d(1, 8, kernel_size=5)
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_channels=8, out_channels=8, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, num_layers=2),
            ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, num_layers=2),
        )

    def forward(self, depth):
        x = self.input_layer(depth)
        out = self.conv_layers(x)
        return out


class CNNTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, depth=4, mlp_ratio=4.0):
        super(CNNTransformer, self).__init__()
        self.rgb_extractor = RGBFeatureExtractor()
        self.depth_extractor = DepthFeatureExtractor()

        self.pos_emb = nn.Parameter(torch.zeros(1, 8 * 10 + 8 * 10, embed_dim))
        self.speed_emb = nn.Linear(1, embed_dim)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim), activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=depth)
        self.ln = nn.LayerNorm(embed_dim)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.regression_head = nn.Sequential(
            nn.Linear(72, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, rgb, depth, hlc, speed, light):
        rgb_features = self.rgb_extractor(rgb)
        depth_features = self.depth_extractor(depth)

        rgb_bs, rgb_c, rgb_h, rgb_w = rgb_features.size()
        rgb_features_reshaped = rgb_features.reshape(rgb_bs, rgb_c, rgb_h * rgb_w).transpose(1, 2)

        depth_bs, depth_c, depth_h, depth_w = depth_features.size()
        depth_features_reshaped = depth_features.reshape(depth_bs, depth_c, depth_h * depth_w).transpose(1, 2)

        transformer_features = torch.cat((rgb_features_reshaped, depth_features_reshaped), dim=1)

        transformer_features = transformer_features + self.pos_emb
        transformer_features += self.speed_emb(speed).unsqueeze(1)
        transformer_output = self.transformer(transformer_features)
        transformer_output = self.ln(transformer_output)

        rgb_features_out = transformer_output[:, :rgb_h * rgb_w, :].transpose(1, 2).reshape(rgb_bs, rgb_c, rgb_h, rgb_w)
        depth_features_out = transformer_output[:, depth_h * depth_w:, :].transpose(1, 2).reshape(depth_bs, depth_c, depth_h, depth_w)

        rgb_features = rgb_features + rgb_features_out
        depth_features = depth_features + depth_features_out

        rgb_features = self.global_pool(rgb_features)
        rgb_features = torch.flatten(rgb_features, 1)

        depth_features = self.global_pool(depth_features)
        depth_features = torch.flatten(depth_features, 1)

        combined_features = rgb_features + depth_features

        hlc = torch.flatten(hlc, 1)
        light = torch.flatten(light, 1)

        x = torch.cat((combined_features, hlc, light), dim=1)

        x = self.regression_head(x)
        out = torch.sigmoid(x)
        return out