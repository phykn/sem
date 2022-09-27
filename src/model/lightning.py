import torch
import pytorch_lightning as pl
from torch.nn import Module
from .scheduler import CosineAnnealingWarmupRestarts


class Lightning(pl.LightningModule):
    def __init__(
        self,
        model: Module,
        scheduler: str="step",
        first_cycle_steps: int=100,
        step_size: int=80,
        warmup_steps: int=10,
        optimizer: str="sgd",
        max_lr: float=0.01,
        min_lr: float=0.001,
        momentum: float=0.9,
        weight_decay: float=1e-4,
        nesterov: bool=True,
        cycle_mult: float=1.0,
        gamma: float=1.0

    ) -> None:
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.weight_decay = weight_decay     
        self.momentum = momentum
        self.nesterov = nesterov
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_size = step_size
        self.warmup_steps = warmup_steps
        self.gamma = gamma   

    def forward(self, data: dict):
        return self.model.predict(data)

    def training_step(self, batch, batch_idx):
        loss = self.model.loss(batch)
        for key in loss.keys():
            self.log(
                name=f"train_{key}", 
                value=loss[key]
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        return self.model.loss(batch)

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            self.log(
                name=f"valid_{key}", 
                value=torch.stack([output[key] for output in outputs]).mean()
            )

    def configure_optimizers(self):
        if self.optimizer=="sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.max_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov
            )
        elif self.optimizer=="adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.max_lr,
                betas=(self.momentum, 0.999),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"{self.optimizer} optimizer is not supported.")

        if self.scheduler=="cosine":
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer, 
                first_cycle_steps=self.first_cycle_steps,
                cycle_mult=self.cycle_mult,
                max_lr=self.max_lr,
                min_lr=self.min_lr,
                warmup_steps=self.warmup_steps,
                gamma=self.gamma
            )
        elif self.scheduler=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.step_size, 
                gamma=self.gamma
            )
        return [optimizer], [{"scheduler": scheduler}]
    
    
def build_lightning(model, args):
    return Lightning(
        model=model,
        scheduler=args.scheduler,
        first_cycle_steps=args.epoch,
        step_size=args.step_size if hasattr(args, "step_size") else int(args.epoch*0.8),
        warmup_steps=args.warmup_steps if hasattr(args, "warmup_steps") else args.epoch//10,
        optimizer=args.optimizer,
        max_lr=args.max_lr,
        min_lr=args.min_lr if hasattr(args, "min_lr") else args.max_lr/10,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov if hasattr(args, "nesterov") else True,
        cycle_mult=args.cycle_mult if hasattr(args, "cycle_mult") else 1.0,
        gamma=args.gamma if hasattr(args, "gamma") else 0.1
    )