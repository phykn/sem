import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        smooth: float=1
    ) -> torch.Tensor:
        # bce
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # dice
        x = inputs.sigmoid().view(-1)
        y = targets.view(-1)
        intersection = (x*y).sum()                            
        dice_loss = 1 - (2.*intersection+smooth)/(x.sum()+y.sum()+smooth)  

        return bce_loss+dice_loss