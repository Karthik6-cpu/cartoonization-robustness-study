import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1)
        )
        self.spatial = nn.Conv2d(
            2, 1, kernel_size=7, padding=3
        )
    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        channel_attention = torch.sigmoid(
            self.channel_mlp(avg) +
            self.channel_mlp(mx)
        )
        x = x * channel_attention
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        spatial_attention = torch.sigmoid(
            self.spatial(torch.cat([avg_map, max_map], dim=1))
        )
        return x * spatial_attention
class SemanticScaleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale_scorer = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels, 3)
        )
    def forward(self, features):
        pooled = [
            F.adaptive_avg_pool2d(f, 1).flatten(1)
            for f in features
        ]
        descriptor = torch.cat(pooled, dim=1)
        weights = torch.softmax(
            self.scale_scorer(descriptor), dim=1
        )
        target_size = features[-1].shape[-2:]
        aligned = [
            f if f.shape[-2:] == target_size
            else F.interpolate(
                f, size=target_size,
                mode="bilinear", align_corners=False
            )
            for f in features
        ]
        fused = sum(
            aligned[i] * weights[:, i].view(-1, 1, 1, 1)
            for i in range(3)
        )
        return fused, weights
class EdgeRefinement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.edge_encoder = nn.Sequential(
            ConvBNAct(1, channels, 3),
            ConvBNAct(channels, channels, 3)
        )
        # Bounded residual contribution.
        self.edge_strength = nn.Parameter(torch.tensor(-2.0))
    def forward(self, image, features):
        # The encoder input is ImageNet-normalized. Recover RGB in [0, 1]
        # before applying luminance weights and fixed Sobel filters.
        mean = image.new_tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)
        std = image.new_tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)
        rgb = (image * std + mean).clamp(0, 1)
        gray = (
            0.299 * rgb[:, 0:1] +
            0.587 * rgb[:, 1:2] +
            0.114 * rgb[:, 2:3]
        )
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        edge = F.interpolate(
            edge,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        edge_features = self.edge_encoder(edge)
        strength = 0.25 * torch.sigmoid(self.edge_strength)
        return features + strength * edge_features
class CAFRNetV2(nn.Module):
    def __init__(
        self,
        num_classes=10,
        pretrained=True,
        dropout=0.25,
        projection_dim=256,
        use_edge=True,
        use_semantic_fusion=True,
        use_attention=True
    ):
        super().__init__()
        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained else None
        )
        backbone = convnext_tiny(weights=weights)
        self.encoder = backbone.features
        # ConvNeXt-Tiny feature stages:
        # 3 -> 192 channels
        # 5 -> 384 channels
        # 7 -> 768 channels
        self.stage_indices = (3, 5, 7)
        self.projections = nn.ModuleList([
            ConvBNAct(192, projection_dim),
            ConvBNAct(384, projection_dim),
            ConvBNAct(768, projection_dim)
        ])
        self.use_edge = use_edge
        self.use_semantic_fusion = use_semantic_fusion
        self.use_attention = use_attention
        self.fusion = SemanticScaleFusion(projection_dim)
        self.edge_refinement = EdgeRefinement(projection_dim)
        self.attention = ChannelSpatialAttention(projection_dim)
        self.refinement = nn.Sequential(
            ConvBNAct(projection_dim, projection_dim, 3),
            nn.Dropout2d(0.1),
            ConvBNAct(projection_dim, projection_dim, 3)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.embedding_dim = projection_dim
    def forward(self, x, return_features=False):
        input_image = x
        stage_features = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in self.stage_indices:
                stage_features.append(x)
        projected = [
            projection(feature)
            for projection, feature
            in zip(self.projections, stage_features)
        ]
        if self.use_semantic_fusion:
            fused, scale_weights = self.fusion(projected)
        else:
            target_size = projected[-1].shape[-2:]
            aligned = [
                f if f.shape[-2:] == target_size else
                F.interpolate(
                    f, size=target_size,
                    mode="bilinear", align_corners=False
                )
                for f in projected
            ]
            scale_weights = torch.full(
                (x.shape[0], 3), 1.0 / 3.0,
                device=x.device, dtype=x.dtype
            )
            fused = sum(aligned) / 3.0
        if self.use_edge:
            fused = self.edge_refinement(input_image, fused)
        if self.use_attention:
            attended = self.attention(fused)
        else:
            attended = fused
        refined = attended + self.refinement(attended)
        embedding = F.adaptive_avg_pool2d(
            refined, 1
        ).flatten(1)
        logits = self.classifier(embedding)
        if return_features:
            return {
                "logits": logits,
                "embedding": embedding,
                "gates": scale_weights,
                "scale_weights": scale_weights
            }
        return logits
def paired_consistency_loss(
    logits_a,
    logits_b,
    temperature=2.0
):
    log_p = F.log_softmax(
        logits_a / temperature, dim=1
    )
    log_q = F.log_softmax(
        logits_b / temperature, dim=1
    )
    p = log_p.exp()
    q = log_q.exp()
    loss_ab = F.kl_div(
        log_p, q, reduction="batchmean"
    )
    loss_ba = F.kl_div(
        log_q, p, reduction="batchmean"
    )
    return (
        0.5 * (loss_ab + loss_ba)
        * temperature ** 2
    )
# Compatibility name for existing imports.
CAFRNet = CAFRNetV2
def build_model(num_classes=10, pretrained=True, dropout=0.25):
    return CAFRNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
