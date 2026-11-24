"""CAFR-Net: Cartoon-Aware Feature Refinement Network.

Research prototype for semantic classification under cartoonization and
compression shifts. The model combines a pretrained ConvNeXt-Tiny encoder,
multi-scale semantic features, an explicit edge-structure stream, and
frequency-conditioned residual fusion.

This is a proposed architecture, not a validated performance claim.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class EdgeStructureBranch(nn.Module):
    """Extracts progressively downsampled edge/contour representations."""

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        channels = (3, 32, 64, 128, 192, out_channels)
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.GELU(),
            ])
        self.encoder = nn.Sequential(*layers)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: Tensor) -> Tensor:
        # Edge extraction uses RGB luminance; input is expected in [0, 1].
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        edge_rgb = magnitude.repeat(1, 3, 1, 1)
        return self.refine(self.encoder(edge_rgb))


class CAFRNet(nn.Module):
    """Cartoon-Aware Feature Refinement Network.

    Args:
        num_classes: Number of target classes.
        pretrained: Load ImageNet-1K ConvNeXt-Tiny weights.
        dropout: Dropout probability before the classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.encoder = convnext_tiny(weights=weights).features
        self.feature_channels = (192, 384, 768)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.GELU(),
            )
            for ch in self.feature_channels
        ])
        self.edge_branch = EdgeStructureBranch(out_channels=256)
        self.frequency_gate = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
        self._initialize_new_layers()

    def _initialize_new_layers(self) -> None:
        for module in (self.projections, self.edge_branch, self.fusion_refine,
                       self.frequency_gate, self.classifier):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _encode_multiscale(self, x: Tensor):
        # torchvision ConvNeXt-Tiny feature indices:
        # 1 -> 96 channels, 3 -> 192, 5 -> 384, 7 -> 768.
        outputs: Dict[int, Tensor] = {}
        z = x
        for index, layer in enumerate(self.encoder):
            z = layer(z)
            if index in (3, 5, 7):
                outputs[index] = z
        return outputs[3], outputs[5], outputs[7]

    def forward(self, x: Tensor, return_features: bool = False):
        # The edge branch consumes denormalized [0,1] RGB. The encoder consumes
        # ImageNet-normalized RGB. The training/evaluation pipeline supplies
        # normalized x; invert normalization for the edge stream.
        mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        rgb = (x * std + mean).clamp(0.0, 1.0)

        f2, f3, f4 = self._encode_multiscale(x)
        target_size = f4.shape[-2:]
        semantic = [
            F.interpolate(project(feature), size=target_size,
                          mode="bilinear", align_corners=False)
            for project, feature in zip(self.projections, (f2, f3, f4))
        ]
        edge = self.edge_branch(rgb)
        edge = F.interpolate(edge, size=target_size, mode="bilinear",
                             align_corners=False)

        # A global gradient-energy descriptor controls the contribution of
        # shallow/deep semantic and edge representations.
        gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2]
                + 0.114 * rgb[:, 2:3])
        gx = F.conv2d(gray, self.edge_branch.sobel_x, padding=1)
        gy = F.conv2d(gray, self.edge_branch.sobel_y, padding=1)
        frequency_energy = torch.sqrt(gx.square() + gy.square() + 1e-6).mean(
            dim=(2, 3)
        )
        gates = torch.softmax(self.frequency_gate(frequency_energy), dim=1)
        streams = semantic + [edge]
        fused = sum(stream * gates[:, i:i + 1, None, None]
                    for i, stream in enumerate(streams))
        fused = fused + self.fusion_refine(fused)
        pooled = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        embedding = self.norm(pooled)
        logits = self.classifier(self.dropout(embedding))

        if return_features:
            return {"logits": logits, "embedding": embedding,
                    "gates": gates, "feature_map": fused}
        return logits


def paired_consistency_loss(
    logits_a: Tensor,
    logits_b: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    """Symmetric KL consistency for matched clean/corrupted training views."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_pa = F.log_softmax(logits_a / temperature, dim=1)
    log_pb = F.log_softmax(logits_b / temperature, dim=1)
    pa, pb = log_pa.exp(), log_pb.exp()
    loss_ab = F.kl_div(log_pa, pb.detach(), reduction="batchmean")
    loss_ba = F.kl_div(log_pb, pa.detach(), reduction="batchmean")
    return 0.5 * (loss_ab + loss_ba) * (temperature ** 2)


def build_model(
    num_classes: int = 10,
    pretrained: bool = True,
    dropout: float = 0.25,
) -> CAFRNet:
    return CAFRNet(num_classes=num_classes, pretrained=pretrained,
                   dropout=dropout)
