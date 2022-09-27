import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A
from typing import List, Dict
from torch.utils.data import Dataset


class AvgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int=112,
        interpolation: int=2,
        data_size: int=10000,
        train: bool=False
    ) -> None:
        """
        interpolation
            0: cv2.INTER_NEAREST
            1: cv2.INTER_LINEAR
            2: cv2.INTER_CUBIC
            3: cv2.INTER_AREA
            4: cv2.INTER_LANCZOS4
        """
        self.files = df["file"].tolist()
        self.depth_avgs = df["depth_avg"].tolist()
        self.data_size = data_size
        self.train = train

        self.transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=interpolation, p=1.0),
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5)
        ])
        self.noise = A.Compose([
            A.Blur(blur_limit=(3, 7), p=0.5),
            A.Downscale(scale_min=0.25, scale_max=0.75, p=0.5),
            A.GaussNoise(var_limit=(10, 50), mean=0, p=0.5)
        ])

    def __len__(
        self
    ) -> int:
        data_len = len(self.files)
        if data_len < self.data_size:
            return data_len
        else:
            return self.data_size

    @staticmethod
    def norm_image(
        x: np.ndarray
    ) -> np.ndarray:
        return x / 255.0

    @staticmethod
    def norm_depth_avg(
        x: float
    ) -> float:
        mean = 123.42
        std = 11.09
        return (x-mean)/std

    def __getitem__(
        self,
        index: int
    ) -> Dict[str, np.ndarray]:
        index = random.randrange(len(self.files))

        # read data
        image = cv2.imread(self.files[index], 0)
        depth_avg = self.depth_avgs[index]

        # transform
        image = self.transform(image=image)["image"]

        # add noise
        if self.train:
            image = self.noise(image=image)["image"]

        # norm
        image = self.norm_image(image)
        depth_avg = self.norm_depth_avg(depth_avg)

        # add axis
        image = image[None, :, :]
        depth = np.zeros_like(image)

        return dict(
            mask=np.array(0, dtype=int),
            label=np.array(0, dtype=int),
            depth_avg=np.array([depth_avg], dtype=float),
            image=image.astype(float),
            depth=depth.astype(float)
        )
    
    
class SimulationDataset(Dataset):
    def __init__(
        self,
        sem_files: List[str],
        depth_files: List[str],
        img_size: int=112,
        interpolation: int=2,
        data_size: int=10000,
        train: bool=False
    ) -> None:
        """
        interpolation
            0: cv2.INTER_NEAREST
            1: cv2.INTER_LINEAR
            2: cv2.INTER_CUBIC
            3: cv2.INTER_AREA
            4: cv2.INTER_LANCZOS4
        """    
        self.sem_files = sem_files
        self.depth_files = depth_files
        self.data_size = data_size
        self.train = train
        
        self.transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=interpolation, p=1.0),
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5)
        ], additional_targets=dict(image_1="image"))
        self.noise = A.Compose([
            A.Blur(blur_limit=(3, 7), p=0.5),
            A.Downscale(scale_min=0.25, scale_max=0.75, p=0.5),
            A.GaussNoise(var_limit=(10, 50), mean=0, p=0.5)
        ])

    def __len__(
        self
    ) -> int:
        data_len = len(self.sem_files)
        if data_len < self.data_size:
            return data_len
        else:
            return self.data_size

    @staticmethod
    def norm_image(
        x: np.ndarray
    ) -> np.ndarray:
        return x / 255.0

    @staticmethod
    def get_label(
        path: str
    ) -> int:
        if "Case_1" in path:
            return 1
        elif "Case_2" in path:
            return 2
        elif "Case_3" in path:
            return 3
        elif "Case_4" in path:
            return 4
        else:
            return 0

    def __getitem__(
        self,
        index: int
    ) -> Dict[str, np.ndarray]:
        index = random.randrange(len(self.sem_files))

        # read data
        image = cv2.imread(self.sem_files[index], 0)
        depth = cv2.imread(self.depth_files[index], 0)

        # transform
        transformed = self.transform(image=image, image_1=depth)
        image = transformed["image"]
        depth = transformed["image_1"]

        # noise
        if self.train:
            image = self.noise(image=image)["image"]

        # normalize
        image = self.norm_image(image)
        depth = self.norm_image(depth)

        # add axis
        image = image[None, :, :]
        depth = depth[None, :, :]

        # label
        label = self.get_label(self.depth_files[index])

        return dict(
            mask=np.array(1, dtype=int),
            label=np.array(label, dtype=int),
            depth_avg=np.array([0], dtype=float),
            image=image.astype(float),
            depth=depth.astype(float)
        )