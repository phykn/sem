import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from timm.models.layers import trunc_normal_


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ListLayerNorm(nn.Module):
    def __init__(
        self,
        channels: List[int]=[96, 192, 384, 768]
    ) -> None:
        super().__init__()
        norms = [LayerNorm(channel, data_format="channels_first") for channel in channels]
        self.norms = nn.Sequential(*norms)

    def forward(
        self,
        xs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return [norm(x) for norm, x in zip(self.norms, xs)]


class UpscaleBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        upsample: str="nearest+conv",
        norm: bool=False,
        act: Optional[str]=None
    ) -> None:
        super().__init__()
        self.upsample = upsample
        layers = []

        # conv
        if upsample == "nearest+conv":
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        elif upsample == "pixelshuffle":
            layers.append(nn.Conv2d(in_channels, 4*out_channels, 3, 1, 1))
            layers.append(nn.PixelShuffle(2))
        elif upsample == "convtranspose":
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        else:
            raise ValueError(
                f"upsample {upsample} is not supported. ('nearest+conv', 'pixelshuffle', 'convtranspose')"
            )

        # norm
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))

        # act
        if act is None:
            pass
        elif act == "lrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif act == "gelu":
            layers.append(nn.GELU())
        elif act == "tanh":
            layers.append(nn.Tanh())
        elif act == "sigmoid":
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"act {act} is not supported. ('lrelu', 'gelu', 'tanh', 'sigmoid')")

        self.layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(
        self, 
        m: nn.Module
    ) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        if self.upsample == "nearest+conv":
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.layers(x)


class Reconstructor(nn.Module):
    def __init__(
        self,
        channels: List[int],
        inter_dim: int=32,
        out_dim: int=1,
        upsample: str="convtranspose",
        norm: bool=True,
        act: str="lrelu"
    ) -> None:
        super().__init__()
        channels = channels[::-1]
        stem = [UpscaleBlock(channels[i], channels[i+1], upsample, norm, act) for i in range(len(channels)-1)]
        head = [
            UpscaleBlock(channels[-1], inter_dim, upsample, norm, act),
            UpscaleBlock(inter_dim, out_dim, upsample, norm, "sigmoid")
        ]
        self.stem = nn.Sequential(*stem)
        self.head = nn.Sequential(*head)

    def forward(
        self,
        xs: List[torch.Tensor]
    ) -> torch.Tensor:
        xs = xs[::-1]
        y = torch.zeros_like(xs[0])
        for layer, x in zip(self.stem, xs):
            y = layer(x+y)
        return self.head(y)


class SpatialGatingUnit(nn.Module):
    def __init__(
        self,
        dim: int,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, 2*dim)
        trunc_normal_(self.linear_1.weight, std=.02)
        nn.init.zeros_(self.linear_1.bias)

        self.linear_2 = nn.Linear(dim, dim)
        trunc_normal_(self.linear_2.weight, std=.02)
        nn.init.zeros_(self.linear_2.bias)

        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)

        res, gate = torch.chunk(x, 2, -1)
        gate = self.norm(gate)
        gate = self.proj(gate)
        x = res * gate

        x = self.linear_2(x)
        return x


class Head(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: bool=False
    ) -> None:
        super().__init__()
        
        self.norm = norm
        if self.norm:
            self.norm_layer = nn.LayerNorm(in_features)

        self.head = nn.Linear(in_features, out_features)
        trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if self.norm:
            x = self.norm_layer(x)
        return self.head(x)