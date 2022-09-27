import os
import cv2
import numpy as np
import torch
from glob import glob
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader


def make_data(
    path: str,
    img_size: int=224,
    interpolation: int=2
) -> Dict[str, np.ndarray]:
    image = cv2.imread(path, 0).astype(float)
    image = cv2.resize(image, dsize=(img_size, img_size), interpolation=interpolation)
    image = image / 255.0
    image = image[None, :, :]
    return dict(
        path=path,
        image=image
    )


def make_single_data(
    path: str,
    img_size: int=224,
    interpolation: int=2,
    device: str="cpu"
) -> Dict[str, torch.Tensor]:
    data = make_data(path, img_size, interpolation)
    data["image"] = torch.from_numpy(data["image"]).unsqueeze(0).to(device)
    return data


class TestDataset(Dataset):
    def __init__(
        self,
        files: List[str]
    ) -> None:
        self.files = files

    def __len__(
        self
    ) -> int:
        return len(self.files)

    def __getitem__(
        self,
        index: int
    ) -> Dict[str, np.ndarray]:
        file = self.files[index]
        return make_data(file)


def build_test_loader(args):
    return DataLoader(
        dataset=TestDataset(glob(os.path.join(args.data_root, "test/SEM/*.png"))),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        shuffle=False,
        drop_last=False
    )