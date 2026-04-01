"""Model builders shared by training and backend inference."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torchvision.models as models

ARCHITECTURE_DISPLAY_NAMES = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "densenet121": "DenseNet121",
}


def default_normalization(grayscale: bool) -> tuple[list[float], list[float]]:
    if grayscale:
        return [0.5], [0.25]
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _make_input_conv(old_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups if in_channels == old_conv.in_channels else 1,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )
    with torch.no_grad():
        if old_conv.weight.shape[1] == in_channels:
            new_conv.weight.copy_(old_conv.weight)
        elif in_channels == 1:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            repeated = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            new_conv.weight.copy_(repeated / max(in_channels, 1))
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv


def adapt_input_conv(model: nn.Module, architecture: str, in_channels: int) -> nn.Module:
    if in_channels == 3:
        return model

    if architecture == "convnext_tiny":
        model.features[0][0] = _make_input_conv(model.features[0][0], in_channels)
        return model

    if architecture == "efficientnet_v2_s":
        model.features[0][0] = _make_input_conv(model.features[0][0], in_channels)
        return model

    if architecture == "densenet121":
        model.features.conv0 = _make_input_conv(model.features.conv0, in_channels)
        return model

    raise ValueError(f"Unsupported architecture for input adaptation: {architecture}")


def build_model(
    architecture: str,
    num_classes: int,
    *,
    in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    if architecture == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model = adapt_input_conv(model, architecture, in_channels)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if architecture == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model = adapt_input_conv(model, architecture, in_channels)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if architecture == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model = adapt_input_conv(model, architecture, in_channels)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")


def get_cam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "convnext_tiny":
        return model.features[-1]
    if architecture == "efficientnet_v2_s":
        return model.features[-1]
    if architecture == "densenet121":
        return model.features
    raise ValueError(f"Unsupported architecture for Grad-CAM: {architecture}")


def display_name(architecture: str) -> str:
    return ARCHITECTURE_DISPLAY_NAMES.get(architecture, architecture)


def normalize_weights(weights: Iterable[float]) -> list[float]:
    weights = [float(w) for w in weights]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]
