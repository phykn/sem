import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from timm.models.layers import trunc_normal_
from .positional_embedding import PositionEmbedding2D
from .transformer import Transformer
from .backbone import Backbone
from .layers import UpscaleBlock, ListLayerNorm, Reconstructor, SpatialGatingUnit, Head


class SEM_model(nn.Module):
    def __init__(
        self,
        backbone: str="resnet18",
        in_chans: int=1,
        depth: int=4,
        pretrained: bool=False,
        upsample: str="convtranspose",
        n_query: int=7,
        d_model: int=128,
        n_head: int=8,
        d_ff: int=1024,
        num_encoder_layers: int=3,
        num_decoder_layers: int=3,
        dropout: float=0.1
    ) -> None:
        super().__init__()
        # backbone
        self.backbone = Backbone(backbone, in_chans, depth, pretrained)
        dims = self.backbone.dims
        self.norms = ListLayerNorm(dims)

        # reconstruct        
        self.recon = Reconstructor(dims, 32, 1, upsample, True, "lrelu")

        # transformer
        self.junction_x = nn.Conv2d(dims[depth-1], d_model, kernel_size=1)
        self.junction_y = nn.Conv2d(dims[depth-1], d_model, kernel_size=1)
        self.pe = PositionEmbedding2D(d_model//2, normalize=True)
        self.query = nn.Embedding(n_query**2, d_model)
        trunc_normal_(self.query.weight, std=.02)

        self.transformer = Transformer(
            d_model=d_model, 
            n_head=n_head, 
            d_ff=d_ff,
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        )

        # depth_avg
        self.mlp = SpatialGatingUnit(d_model)
        self.avg_head = Head(d_model, 1, norm=True)

        # depth_map
        corrector = []
        curr_dim = d_model
        for _ in range(len(dims)):
            upscale = UpscaleBlock(curr_dim, curr_dim//2, upsample, True, "lrelu")
            curr_dim = curr_dim // 2
            corrector.append(upscale)
        corrector += [UpscaleBlock(curr_dim, 1, upsample, True, "sigmoid")]
        self.corrector = nn.Sequential(*corrector)

        # loss func
        self.avg_loss = nn.SmoothL1Loss()
        self.map_loss = nn.L1Loss()

    def init_head(
        self, 
        m: nn.Module
    ) -> None:
        trunc_normal_(m.weight, std=.02)
        nn.init.constant_(m.bias, 0)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # backbone
        xs = self.norms(self.backbone(x))

        # draft
        draft = self.recon(xs)

        # transformer
        x = self.junction_x(xs[-1])
        p = self.pe(x)
        x = x.flatten(2).permute(2, 0, 1)
        p = p.flatten(2).permute(2, 0, 1)

        ys = self.norms(self.backbone(draft))
        y = self.junction_y(ys[-1])
        y = y.flatten(2).permute(2, 0, 1)

        _, batch_size, _ = x.shape
        q = self.query.weight.unsqueeze(1).repeat(1, batch_size, 1)

        tf_memory, tf_output = self.transformer(x, p, y, q)

        # depth_avg
        depth_avg = self.avg_head(self.mlp(tf_memory.mean(-1)))

        # depth_map
        b, c, hw = tf_output.shape
        h = w = int(math.sqrt(hw))
        m = tf_output.reshape(b, c, h, w)
        depth = self.corrector(m)

        return dict(
            depth_avg=depth_avg,
            draft=draft,            
            depth=depth            
        )

    def predict(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self(data["image"].float())
    
    def loss(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # prediction
        pred = self.predict(data)

        # initialize
        dtype, device = pred["depth"].dtype, pred["depth"].device
        if not hasattr(self, "memo_avg_loss"):
            self.memo_avg_loss = torch.tensor(0, dtype=dtype, device=device)
        if not hasattr(self, "memo_draft_loss"):
            self.memo_draft_loss = torch.tensor(0, dtype=dtype, device=device)
        if not hasattr(self, "memo_depth_loss"):
            self.memo_depth_loss = torch.tensor(0, dtype=dtype, device=device)
        if not hasattr(self, "memo_depth_rmse"):
            self.memo_depth_rmse = torch.tensor(0, dtype=dtype, device=device)

        # index
        avg_index = torch.where(data["mask"]<0.5)[0]
        map_index = torch.where(data["mask"]>0.5)[0]

        # depth_avg
        if len(avg_index) > 0:
            true_depth_avg = data["depth_avg"][avg_index].float()
            pred_depth_avg = pred["depth_avg"][avg_index]
            avg_loss = self.avg_loss(pred_depth_avg, true_depth_avg)
            self.memo_avg_loss = avg_loss.detach()
        else:
            avg_loss = self.memo_avg_loss

        # depth_map
        if len(map_index) > 0:
            # map
            true_draft = data["depth"][map_index].float()
            pred_draft = pred["draft"][map_index]
            draft_loss = self.map_loss(pred_draft, true_draft)

            true_depth = data["depth"][map_index].float()
            pred_depth = pred["depth"][map_index]
            depth_loss = self.map_loss(pred_depth, true_depth)

            depth_rmse = torch.sqrt(torch.square(255.0*(true_depth-pred_depth))).mean()

            self.memo_draft_loss = draft_loss.detach()
            self.memo_depth_loss = depth_loss.detach()
            self.memo_depth_rmse = depth_rmse.detach()
        else:
            draft_loss = self.memo_draft_loss
            depth_loss = self.memo_depth_loss
            depth_rmse = self.memo_depth_rmse

        # total_loss
        if len(map_index) == 0:
            total_loss = 0.1*avg_loss
        elif len(avg_index) == 0:
            total_loss = draft_loss + depth_loss
        else:
            total_loss = 0.1*avg_loss + draft_loss + depth_loss
        
        return dict(
            avg_loss=avg_loss,
            draft_loss=draft_loss,
            depth_loss=depth_loss,
            depth_rmse=depth_rmse,
            loss=total_loss
        )