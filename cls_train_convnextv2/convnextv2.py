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

class CustomConvNeXt(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        feature_dim = base_model.head.fc.in_features
        
        self.feature_enhance = nn.Sequential(
            LayerNorm(feature_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            GRN(feature_dim)
        )

        self.head = nn.Sequential(
            LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.feature_enhance.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.feature_enhance(x)
        if len(x.shape) == 4:
            x = x.mean(dim=(2, 3))
        x = self.head(x)
        return x

def create_model_convnext(num_classes=3, pretrained=True, drop_path_rate=0.1):
    base_model = create_model(
        'convnextv2_base',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )
    
    model = CustomConvNeXt(base_model, num_classes)
    return model