from PIL.Image import Image
import numpy as np
import torch
import torch.cuda
import torch.nn
from typing import Optional, Union
from torch.types import Number
from utils import create_circular_mask, transform
from torchvision import transforms


class PatchMaker:
    def __init__(self, mean, std, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cpu")
        mean = torch.tensor(mean, dtype=torch.float)
        std = torch.tensor(std, dtype=torch.float)

        val = lambda x: ((x - mean) / std).to(self.device).unsqueeze(1).unsqueeze(1)
        self.min_val: torch.Tensor = val(0)
        self.max_val: torch.Tensor = val(1)

        self._to_pil = transforms.Compose(
            [
                transforms.Normalize(np.zeros(3), 1 / std),
                transforms.Normalize(-mean, np.ones(3)),
                transforms.Lambda(lambda img: torch.clamp(img, 0, 1)),
                transforms.ToPILImage(),
            ]
        )
        self.set_transforms()
        self.random_init_patch(100)

    @property
    def pil_patch(self) -> Image:
        return self._to_pil(self._patch.data.squeeze().detach().cpu())  # type: ignore

    @property
    def patch(self):
        return self._patch

    @patch.setter
    def patch(self, init_patch: torch.Tensor):
        self.set_patch(init_patch)

    def random_init_patch(self, size: int, mask: Optional[torch.Tensor] = None):
        init_patch = (
            torch.rand([3, size, size], device=self.device)
            * (self.max_val - self.min_val)
            + self.min_val
        )
        return self.set_patch(init_patch, mask=mask)

    def set_patch(self, init_patch: torch.Tensor, size=None, mask=None):
        patch = init_patch.to(self.device)
        dsize = patch.shape[-2:]

        if mask is None:
            # make patch squared
            min_ind = np.argmin(dsize)
            start = int((dsize[min_ind-1] - dsize[min_ind])/2)
            end = dsize[min_ind] + start
            if min_ind == 0:
                patch = patch[:, :, start:end]
            else:
                patch = patch[:, start:end, :]

            dsize = patch.shape[-2:]
            mask = torch.tensor(create_circular_mask(*dsize))

        patch = patch.unsqueeze(0)
        mask = mask.to(torch.float).to(self.device)
        mask = mask.expand(1, 1, -1, -1)

        if size is not None:
            scale = size / min(dsize)
            dsize = (np.array(dsize) * scale).astype(int)
            patch = transform(patch, 0, scale, dsize=dsize, device=self.device)
            mask = transform(mask, 0, scale, dsize=dsize, device=self.device)

        patch = patch * mask + (1 - mask) * self.max_val.expand(3, *dsize)
        self._patch = torch.nn.Parameter(patch, requires_grad=True)
        self._patch_size = torch.tensor(self._patch.shape[-2:], device=self.device)
        self.mask = (mask != 0).to(torch.float)

    def clamp(self, image_tensor: torch.Tensor):
        return image_tensor.min(other=self.max_val).max(other=self.min_val)

    def applicate_patch(
        self,
        img_tensor: torch.Tensor,
        angle: Union[torch.Tensor, Number],
        scale: Union[torch.Tensor, Number],
        shear: Optional[torch.Tensor] = None,
        location: Optional[torch.Tensor] = None,
    ):
        bs = img_tensor.shape[0]
        mask = transform(
            self.mask.expand(bs, -1, -1, -1),
            angle,
            scale,
            shear,
            location,
            img_tensor.shape[-2:],
            device=self.device,
        )
        mask = (mask != 0).to(torch.float)
        patch = self.clamp(self._patch)
        patch = transform(
            patch.expand(bs, -1, -1, -1),
            angle,
            scale,
            shear,
            location,
            img_tensor.shape[-2:],
            device=self.device,
        )

        applied = img_tensor * (1 - mask) + patch * mask
        return self.clamp(applied)

    def set_transforms(
        self, rotate_angle=(-90, 90), shear=(0, 0), size_by_im=(0.2, 0.45)
    ):
        gen_rand = lambda k, a, b: torch.rand(k, device=self.device) * (b - a) + a
        self.tr_rotate_angle = lambda k: gen_rand(k, *rotate_angle)
        self.tr_shear = lambda k: gen_rand(k, *shear).unsqueeze(1).expand(-1, 2)
        self.tr_size_by_im = lambda k: gen_rand(k, *size_by_im)

    def random_patch_place(self, img_tensor: torch.Tensor):
        b = img_tensor.shape[0]
        rotation = self.tr_rotate_angle(b)
        shear = self.tr_shear(b)
        size_by_im = self.tr_size_by_im(b)

        im_size = torch.tensor(img_tensor.shape[-2:], device=self.device)

        side_size = min(im_size) * (size_by_im * 2 / np.pi).sqrt()  # type: ignore

        scale = side_size / min(self._patch_size)

        location = torch.rand(b, 2, device=self.device) * (
            im_size - scale.unsqueeze(1) * self._patch_size.unsqueeze(0)
        )

        return self.applicate_patch(img_tensor, rotation, scale, shear, location)
