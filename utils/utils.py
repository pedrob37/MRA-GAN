import os
from collections import OrderedDict
import nibabel as nib

from typing import Dict, Hashable, Mapping, Sequence, Union

import numpy as np
import torch

from monai.config import KeysCollection
from monai.transforms.transform import MapTransform, Transform
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
)

GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_dir(path):              # if folder does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)


def new_state_dict(file_name):
    state_dict = torch.load(file_name)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def create_path(some_path):
    if not os.path.exists(some_path):
        os.makedirs(some_path)


def save_img(image, affine, filename):
    nifti_img = nib.Nifti1Image(image, affine)
    if os.path.exists(filename):
        raise OSError("File already exists! Killing job")
    else:
        nib.save(nifti_img, filename)


class CoordConv(Transform):
    """
    Appends additional channels encoding coordinates of the input.
    Liu, R. et al. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, NeurIPS 2018.
    """

    def __init__(
        self,
        spatial_channels: Sequence[int],
    ) -> None:
        """
        Args:
            spatial_channels: the spatial dimensions that are to have their coordinates encoded in a channel and
                appended to the input. E.g., `(1,2,3)` will append three channels to the input, encoding the
                coordinates of the input's three spatial dimensions (0 is reserved for the channel dimension).
        """
        self.spatial_channels = spatial_channels

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: data to be transformed, assuming `img` is channel first.
        """
        if max(self.spatial_channels) > img.ndim - 1:
            raise ValueError(
                f"input has {img.ndim-1} spatial dimensions, cannot add CoordConv channel for dim {max(self.spatial_channels)}."
            )
        if 0 in self.spatial_channels:
            raise ValueError("cannot add CoordConv channel for dimension 0, as 0 is channel dim.")

        # Correction
        batch_size_shape, dim_x, dim_y, dim_z = img.shape
        xx_range = torch.arange(dim_x, dtype=torch.int32) * (1 / (dim_x - 1))
        xx_range = xx_range * 2 - 1
        xx_range = xx_range[None, :, None, None]
        xx_channel = xx_range.repeat(1, 1, dim_y, dim_z)

        yy_range = torch.arange(dim_y, dtype=torch.int32) * (1 / (dim_x - 1))
        yy_range = yy_range * 2 - 1
        yy_range = yy_range[None, None, :, None]
        yy_channel = yy_range.repeat(1, dim_x, 1, dim_z)

        zz_range = torch.arange(dim_z, dtype=torch.int32) * (1 / (dim_x - 1))
        zz_range = zz_range * 2 - 1
        zz_range = zz_range[None, None, None, :]
        zz_channel = zz_range.repeat(1, dim_x, dim_y, 1)

        coord_channels = torch.cat([xx_channel, yy_channel, zz_channel], dim=0).numpy()

        return coord_channels


class CoordConvd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CoordConv`.
    """

    def __init__(self, keys: KeysCollection, spatial_channels: Sequence[int], allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
            spatial_channels: the spatial dimensions that are to have their coordinates encoded in a channel and
                appended to the input. E.g., `(1,2,3)` will append three channels to the input, encoding the
                coordinates of the input's three spatial dimensions. It is assumed dimension 0 is the channel.
        """
        super().__init__(keys, allow_missing_keys)
        self.coord_conv = CoordConv(spatial_channels)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            d["coords"] = self.coord_conv(d[key])
        return d
