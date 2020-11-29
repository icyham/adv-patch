from typing import Literal, Optional, Tuple, Union
import numpy as np
from torch.types import Number
import torchgeometry as tgm
import torch

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)  # type: ignore

    mask = dist_from_center <= radius
    return mask


def transform(
    img_tensor: torch.Tensor,
    angle: Union[torch.Tensor, Number],
    scale: Union[torch.Tensor, Number],
    shear: Optional[torch.Tensor]=None,
    location: Optional[torch.Tensor]=None,
    dsize: Optional[torch.Size] = None,
    device: torch.device = torch.device("cpu"),
):
    img_tensor = img_tensor.to(device)
    while img_tensor.dim() < 4:
        img_tensor = img_tensor.unsqueeze(0)
    b = img_tensor.shape[0]
    angle = torch.zeros(b, device=device) + angle
    scale = torch.zeros(b, device=device) + scale
    center = torch.zeros(b, 2, device=device)
    center[..., 0] = img_tensor.shape[3] / 2  # x
    center[..., 1] = img_tensor.shape[2] / 2  # y

    M = tgm.get_rotation_matrix2d(center, angle, scale)
    if location is None:
        location = torch.zeros(2, device=device)
    location = (scale - 1).unsqueeze(1) * center + location
    M.T[2] += location.T
    if shear is None:
        shear = torch.zeros(2, device=device)
    sm = torch.zeros(b, 2, 3, device=device)
    sm[..., 0, 1] = shear[..., 0]
    sm[..., 1, 0] = shear[..., 1]
    M += sm
    if dsize is None:
        dsize = img_tensor.shape[-2:]
    return tgm.warp_affine(img_tensor, M, dsize) # type: ignore
