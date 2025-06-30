import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class GRN(nn.Module):
    """Global Response Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = self.gamma * (x * Nx) + self.beta + x
        return x.permute(0, 3, 1, 2) # [B, C, H, W]

class ConvNeXtSegmentation(nn.Module):
    def __init__(
        self,
        pretrained=True,
        in_chans=3,
        num_classes=1,
        drop_path_rate=0.1,
        img_size=512,
    ):
        super().__init__()
        
        self.backbone = create_model(
            'convnextv2_tiny',
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        
        self.seg_head = nn.Sequential(
            LayerNorm(768, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.GELU(),
            GRN(256),
            # upsampling
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 1/32 -> 1/16
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 1/16 -> 1/8
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 1/8 -> 1/4
            nn.GELU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)  # 1/4 -> 1/2
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.seg_head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.backbone.forward_features(x)  # [B, C=768, H/32, W/32]
        x = self.seg_head(features)  # [B, num_classes, H/2, W/2]
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x 