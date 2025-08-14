import torch
import torch.nn as nn
import torchvision.models as models
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.shared_mlp(self.channel_avg_pool(x))
        max_out = self.shared_mlp(self.channel_max_pool(x))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1)))
        x = x * spatial_attention
        return x

class EfficientNetMultiScaleCBAM_3scales(nn.Module):
    """
    使用 EfficientNetV2-M 將最後三層(Stage6, Stage7, Head Conv)輸出的特徵圖(C=304,512,1280)，
    各自經過 CBAM 後可以做 Global Average Pooling，再寫一方法融合成一個向量，
    送入 MLP head 回歸 keypoints。
    請參考作業模型圖的作法。
    """
    
    def __init__(self, num_points: int):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features # 利用 features 將對應的層數切出
        """
            TODO
        """

    def forward(self, x):
        """
            TODO
        """
        return 

class EfficientNet(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.efficientnet_v2_m(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_points * 2)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)
    

def initialize_model(model_name, num_points):
    if model_name == "efficientnet":
        return EfficientNet(num_points)
    elif model_name == "efficientnet_ms_cbam_3scales":
        return EfficientNetMultiScaleCBAM_3scales(num_points)
    else:
        raise ValueError("Model must be 'efficientnet', 'efficientnet_ms_cbam_3scales'.")