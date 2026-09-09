import random
from typing import Tuple, List, Union
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class SynchronizedRandomRotation:


    def __init__(
        self,
        degrees: Tuple[float, float] = (-30, 30),
        prob: float = 0.8,
        fill: int = 0,
    ):

        if len(degrees) != 2:
            raise ValueError(f"degrees must be a tuple of (min, max), got {degrees}")

        if degrees[0] > degrees[1]:
            raise ValueError(f"degrees[0] must be <= degrees[1], got {degrees}")

        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")

        self.degrees = degrees
        self.prob = prob
        self.fill = fill

    def __call__(
        self,
        degraded: Union[Image.Image, np.ndarray],
        target: Union[Image.Image, np.ndarray],
        segmentation: Union[Image.Image, np.ndarray],
    ) -> Tuple[Union[Image.Image, np.ndarray],
               Union[Image.Image, np.ndarray],
               Union[Image.Image, np.ndarray]]:


        if random.random() > self.prob:

            return degraded, target, segmentation


        angle = random.uniform(self.degrees[0], self.degrees[1])


        rotated_degraded = self._rotate_image(degraded, angle, InterpolationMode.BILINEAR)


        rotated_target = self._rotate_image(target, angle, InterpolationMode.BILINEAR)


        rotated_segmentation = self._rotate_image(segmentation, angle, InterpolationMode.NEAREST)

        return rotated_degraded, rotated_target, rotated_segmentation

    def _rotate_image(
        self,
        img: Union[Image.Image, np.ndarray],
        angle: float,
        interpolation: InterpolationMode,
    ) -> Union[Image.Image, np.ndarray]:


        is_numpy = isinstance(img, np.ndarray)
        if is_numpy:

            pil_img = Image.fromarray(img)
        else:
            pil_img = img


        rotated_pil = F.rotate(
            pil_img,
            angle=angle,
            interpolation=interpolation,
            expand=False,
            fill=self.fill,
        )


        if is_numpy:
            return np.array(rotated_pil)
        else:
            return rotated_pil

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"degrees={self.degrees}, "
            f"prob={self.prob}, "
            f"fill={self.fill})"
        )
