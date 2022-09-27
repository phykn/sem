from dataclasses import dataclass
from typing import Optional


@dataclass
class get_config:
    train_folder: str="trained"
    
    train_sem_folder: str="C:/data/samsung_sem/open/train/SEM"
    simul_sem_folder: str="C:/data/samsung_sem/open/simulation_data/SEM"
    simul_depth_folder: str="C:/data/samsung_sem/open/simulation_data/Depth"
    average_depth_path: str="C:/data/samsung_sem/open/train/average_depth.csv"    
    test_size: float=0.05
    random_state: int=42

    img_size: int=112
    interpolation: int=2
    train_data_size: int=10000
    valid_data_size: int=10000
    batch_size: int=32
    num_workers: int=4
    pin_memory: bool=True
    persistent_workers: bool=True

    backbone: str="convnext_tiny"
    in_chans: int=1
    depth: int=3
    pretrained: bool=True
    upsample: str="convtranspose"
    n_query: int=7
    d_model: int=256
    n_head: int=8
    d_ff: int=1024
    num_encoder_layers: int=3
    num_decoder_layers: int=3
    dropout: float=0.1
    weight: Optional[str]=None
    
    epoch: int=100
    scheduler: str="step"
    step_size: int=80
    warmup_steps: int=10
    optimizer: str="adamw"
    max_lr: float=1e-4
    min_lr: float=1e-5
    momentum: float=0.9
    weight_decay: float=1e-4
    gamma: float=0.1
    
    ckpt_monitor: str="valid_depth_rmse"
    ckpt_mode: str="min"

    ema_decay: float=0.999
    
    gradient_clip_val: float=1.0
    accumulate_grad_batches: int=1
    log_every_n_steps: int=10
    precision: int=16
    accelerator: str="gpu"
    devices: int=1