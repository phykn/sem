import torch
import torch.nn as nn
import torchvision
from .convnext import convnext_tiny


class ResNetEncoder(nn.Module):
    def __init__(
        self, 
        backbone: str="resnet50", 
        depth: int=4,
        in_chans: int=1,
        pretrained: bool=False
    ) -> None:
        assert depth < 5, "depth should be smaller than 5."
        super().__init__()
        if backbone == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            model = torch.hub.load("pytorch/vision:v0.10.0", backbone, weights=weights)
            self.dims = [64, 128, 256, 512][:depth]
        elif backbone == "resnet50":
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            model = torch.hub.load("pytorch/vision:v0.10.0", backbone, weights=weights)
            self.dims = [256, 512, 1024, 2048][:depth]
        
        layers = list(model.children())
        stem = layers[:4]
        if in_chans != 3:
            stem[0] = nn.Conv2d(in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.stem = nn.Sequential(*stem)
        self.stages = nn.Sequential(*layers[4:4+depth])
        self.depth = depth

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.stem(x)
        xs = []
        for i in range(self.depth):
            x = self.stages[i](x)
            xs.append(x)
        return xs


class ConvNextEncoder(nn.Module):
    def __init__(
        self, 
        backbone: str="convnext_tiny", 
        depth: int=4,
        in_chans: int=1,
        pretrained: bool=False
    ) -> None:
        assert depth < 5, "depth should be smaller than 5."
        super().__init__()
        if backbone == "convnext_tiny":
            model = convnext_tiny(pretrained=pretrained)
            self.dims = [96, 192, 384, 768][:depth]
        
        layers = list(model.children())
        downsample_layers = layers[0][:depth]
        if in_chans != 3:
            downsample_layers[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        stages = layers[1][:depth]

        self.downsample_layers = nn.Sequential(*downsample_layers)
        self.stages = nn.Sequential(*stages)
        self.depth = depth

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        xs = []
        for i in range(self.depth):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            xs.append(x)
        return xs


class Backbone(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_chans: int=1,
        depth: int=4,
        pretrained: bool=False
    ) -> None:
        super().__init__()
        if backbone in ["resnet18", "resnet50"]:
            self.backbone = ResNetEncoder(backbone=backbone, in_chans=in_chans, depth=depth, pretrained=pretrained)
        elif backbone in ["convnext_tiny"]:
            self.backbone = ConvNextEncoder(backbone=backbone, in_chans=in_chans, depth=depth, pretrained=pretrained)
        else:
            raise ValueError(f"{backbone} is not supported. ('resnet18', 'resnet50', 'convnext_tiny')")
        self.dims = self.backbone.dims

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone(x)