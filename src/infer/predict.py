import cv2
import numpy as np
import torch
from typing import Dict


DEPTH_AVG_MEAN = 123.42
DEPTH_AVG_STD = 11.09


@torch.no_grad()
def predict(
    model, 
    data: Dict[str, torch.Tensor],
    interpolation: int=2
) -> Dict[str, np.ndarray]:
    # model predict
    p = model.predict(data)

    # preprocess
    p["depth_avg"] = p["depth_avg"].flatten()
    p["draft"] = torch.clip(p["draft"].permute(0, 2, 3, 1)[:, :, :, 0], 0, 1)
    p["depth"] = torch.clip(p["depth"].permute(0, 2, 3, 1)[:, :, :, 0], 0, 1)

    # to numpy
    for key in p.keys():
        p[key] = p[key].cpu().numpy()

    # post process
    p["depth_avg"] = DEPTH_AVG_STD * p["depth_avg"] + DEPTH_AVG_MEAN
    p["draft"] = np.round((p["draft"])*255.0).astype(np.uint8)
    p["depth"] = np.round((p["depth"])*255.0).astype(np.uint8)

    # resize
    p["draft"] = np.array([cv2.resize(draft, dsize=(48, 72), interpolation=interpolation) for draft in p["draft"]])
    p["depth"] = np.array([cv2.resize(depth, dsize=(48, 72), interpolation=interpolation) for depth in p["depth"]])
    return p