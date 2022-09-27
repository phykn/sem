import os
import json
import torch
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from src.config import get_train_config
from src.dataset import build_dataloader
from src.model import build_model, build_lightning, Lightning, EMA
   
    
def main(args):
    # set folder
    dir_root = args.train_folder
    dir_base = os.path.join(dir_root, str(datetime.now())[2:19].replace(":", "-").replace(" ", "-").replace("-", "_"))
    dir_ckpt = os.path.join(dir_base, "checkpoint")
    dir_weight = os.path.join(dir_base, "weight")

    # make dir
    os.makedirs(dir_base, exist_ok=True)
    os.makedirs(dir_ckpt, exist_ok=True)
    os.makedirs(dir_weight, exist_ok=True)

    # save argument
    with open(os.path.join(dir_base, "argument.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
        
    # dataloader
    train_loader, valid_loader = build_dataloader(args)

    # model
    model = build_model(args)
    if args.weight is not None:
        print(f"Load weight: {args.weight}")
        model.load_state_dict(torch.load(args.weight))
    lightning = build_lightning(model, args)

    # checkpoint
    checkpoint = ModelCheckpoint(
        dirpath=dir_ckpt,
        monitor=args.ckpt_monitor,
        mode=args.ckpt_mode,
        filename="checkpoint-{epoch:02d}",
        every_n_epochs=1,
        save_last=True,
        save_weights_only=True,
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=pl_loggers.TensorBoardLogger(save_dir=dir_base, name="log"),
        check_val_every_n_epoch=1,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[
            checkpoint, 
            LearningRateMonitor(logging_interval="step"), 
            EMA(decay=args.ema_decay, pin_memory=args.pin_memory)
        ]
    )

    # train
    trainer.fit(lightning, train_loader, valid_loader)

    # best model save
    best_model = Lightning.load_from_checkpoint(
        checkpoint_path=checkpoint.best_model_path,
        model=model).model
    torch.save(best_model.state_dict(), os.path.join(dir_weight, "best.pt"))

    # last model save
    last_model = Lightning.load_from_checkpoint(
        checkpoint_path=checkpoint.last_model_path,
        model=model).model
    torch.save(last_model.state_dict(), os.path.join(dir_weight, "last.pt"))
    

if __name__ == "__main__":
    main(get_train_config())