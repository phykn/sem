import os
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader
from .depth_avg import readDepthAvg
from .dataset import AvgDataset, SimulationDataset


def depth2sem(path, itr):
    path = path.replace("Depth", "SEM")
    if itr == 0:
        path = path.replace(".png", "_itr0.png")
    elif itr == 1:
        path = path.replace(".png", "_itr1.png")
    else:
        raise ValueError
    return path
    

def build_dataloader(args):
    # train (sem and average depth)
    df = readDepthAvg(args.train_sem_folder, args.average_depth_path)()
    df_train, df_valid = train_test_split(
        df, 
        test_size=args.test_size, 
        random_state=args.random_state,
        shuffle=True
    )
    
    # simulation (sem and depth)
    depth_files = glob(os.path.join(args.simul_depth_folder, "*/*/*.png"))
    sem_0_files = [depth2sem(path, itr=0) for path in depth_files]
    sem_1_files = [depth2sem(path, itr=1) for path in depth_files]

    depth_files = depth_files + depth_files
    sem_files = sem_0_files + sem_1_files

    train_sem_files, valid_sem_files, train_depth_files, valid_depth_files = train_test_split(
        sem_files, depth_files,
        test_size=args.test_size, 
        random_state=args.random_state,
        shuffle=True
    )

    # dataset
    train_dataset = ConcatDataset([
        AvgDataset(
            df=df_train,
            img_size=args.img_size,
            interpolation=args.interpolation,
            data_size=args.train_data_size,
            train=True
        ),
        SimulationDataset(
            sem_files=train_sem_files,
            depth_files=train_depth_files, 
            img_size=args.img_size,
            interpolation=args.interpolation,
            data_size=args.train_data_size*2,
            train=True          
        )
    ])
    valid_dataset = ConcatDataset([
        AvgDataset(
            df=df_valid,
            img_size=args.img_size,
            interpolation=args.interpolation,
            data_size=args.valid_data_size,
            train=False
        ),
        SimulationDataset(
            sem_files=valid_sem_files,
            depth_files=valid_depth_files, 
            img_size=args.img_size,
            interpolation=args.interpolation,
            data_size=args.valid_data_size*2,
            train=False
        )
    ])

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        shuffle=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        shuffle=True,
        drop_last=True
    )    
    return train_loader, valid_loader