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


def create_path(some_dir):
    try:
        if not os.path.exists(some_dir):
            os.makedirs(some_dir)
    except FileExistsError:
        print(f"{some_dir} already exists!")
        pass


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

    @ staticmethod
    def normalise_images(array):
        import numpy as np
        return (array - np.min(array)) / (np.max(array) - np.min(array))


""" Variational Autoencoder

Based on
https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI/blob/master/models/variational_autoencoder.py
"""
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, z_dim=512, linear_in_feats=2800, dropout_rate=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.linear_in_feats = linear_in_feats

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, 5, stride=2, padding=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
        )

        self.intermediate_conv = nn.Conv3d(128, 16, 1)
        self.mu_layer = nn.Linear(self.linear_in_feats, z_dim)
        # self.sigma_layer = nn.Linear(self.linear_in_feats, z_dim)
        # self.dropout_mu_layer_enc = nn.Dropout3d(p=dropout_rate)
        # self.dropout_sigma_layer_enc = nn.Dropout3d(p=dropout_rate)
        #
        # self.dec_dense = nn.Linear(z_dim, self.linear_in_feats)
        # self.dropout_layer_dec = nn.Dropout3d(p=dropout_rate)
        # self.intermediate_conv_reverse = nn.Conv3d(16, 128, 1)
        #
        # self.decoder = nn.Sequential(
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(),
        #     nn.ConvTranspose3d(128, 128, 4, stride=2, padding=1),
        #     nn.BatchNorm3d(128),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
        #     nn.BatchNorm3d(64),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
        #     nn.BatchNorm3d(32),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose3d(32, 32, 4, stride=2, padding=1),
        #     nn.BatchNorm3d(32),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv3d(32, 1, 1),
        # )

    def encode(self, x):
        temp_out = self.encoder(x)

        temp_out = self.intermediate_conv(temp_out)
        flatten = temp_out.view(-1, self.linear_in_feats)

        z_mu = self.mu_layer(flatten)
        # z_log_sigma = self.dropout_sigma_layer_enc(self.sigma_layer(flatten))
        # z_sigma = torch.exp(z_log_sigma)

        # Reshape
        output_spatial_side = int(x.shape[-1] / 16)
        z_mu_reshaped = z_mu.view(1, -1,
                                  output_spatial_side,
                                  output_spatial_side,
                                  output_spatial_side)  # 8 for patch size 128

        return z_mu_reshaped  # z_sigma  # Need to edit assym. script!

    # def sampling(self, z_mu, z_sigma):
    #     eps = torch.randn_like(z_sigma)
    #     z_vae = z_mu + eps * z_sigma
    #     return z_vae
    #
    # def decode(self, z_vae):
    #     reshaped = self.dropout_layer_dec(self.dec_dense(z_vae)).view(-1, 16, 5, 7, 5)
    #     temp_out = self.intermediate_conv_reverse(reshaped)
    #     temp_out = self.decoder(temp_out)
    #
    #     return temp_out

    def forward(self, x):
        # z_mu, z_sigma = self.encode(x)
        # z = self.sampling(z_mu, z_sigma)
        # reconstruction = self.decode(z)
        z = self.encode(x)
        return z

    # def reconstruct(self, x):
    #     z_mu, _ = self.encode(x)
    #     reconstruction = self.decode(z_mu)
    #     return reconstruction

from typing import Callable, Union

import torch
import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple


def custom_sliding_window_inference(
    inputs: Union[torch.Tensor, tuple],
    roi_size,
    sw_batch_size: int,
    predictor: Callable,
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval=0,
    uncertainty_flag=False,
    num_loss_passes=20
):
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0

    Raises:
        NotImplementedError: inputs must have batch_size=1.

    Note:
        - input must be channel-first and have a batch dim, support both spatial 2D and 3D.
        - currently only supports `inputs` with batch_size=1.
    """
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    inputs_type = type(inputs)
    if inputs_type == tuple:
        phys_inputs = inputs[1]
        inputs = inputs[0]
    num_spatial_dims = len(inputs.shape) - 2
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError("inputs must have batch_size=1.")

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    # print(f'The slices are {slices}')

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1], curr_slice[2]])
            else:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1]])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    if not uncertainty_flag:
        # No uncertainty, so only one prediction, so proceed normally
        output_rois = list()
        for data in slice_batches:
            if not uncertainty_flag and inputs_type == tuple:
                # print(f"SWI data shape: {data.shape}")
                seg_prob = predictor(data, phys_inputs)  # batched patch segmentation
                output_rois.append(seg_prob)
            elif inputs_type != tuple:
                try:
                    # print(f"SWI data shape: {data.shape}")
                    seg_prob = predictor(data)  # batched patch segmentation
                    output_rois.append(seg_prob)
                except TypeError:
                    # print(f"SWI data shape: {data.shape}")
                    seg_prob = predictor(data)
                    output_rois.append(seg_prob)
            # print(f"Seg prob length: {len(seg_prob)}")
            # print(f"Inner seg prob shape {len(seg_prob[0])}")
            # print(f"Seg prob: {seg_prob}")
            # print(f"Seg prob shape: {seg_prob[0][0].shape}")

        # stitching output image
        output_classes = output_rois[0].shape[1]
        output_shape = [batch_size, output_classes] + list(image_size)

        # Create importance map
        importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode, device=inputs.device)

        # allocate memory to store the full output and the count for overlapping parts
        output_image = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
        count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

        for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
            slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

            # store the result in the proper location of the full output. Apply weights from importance map.
            for curr_index in slice_index_range:
                curr_slice = slices[curr_index]
                if len(curr_slice) == 3:
                    # print(output_image.shape, curr_slice, importance_map.shape, output_rois[window_id].shape)
                    output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                        importance_map * output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                else:
                    output_image[0, :, curr_slice[0], curr_slice[1]] += (
                        importance_map * output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

        # account for any overlapping sections
        output_image /= count_map

        if num_spatial_dims == 3:
            return output_image[
                ...,
                pad_size[4]: image_size_[0] + pad_size[4],
                pad_size[2]: image_size_[1] + pad_size[2],
                pad_size[0]: image_size_[2] + pad_size[0],
            ]
        return output_image[
            ..., pad_size[2]: image_size_[0] + pad_size[2], pad_size[0]: image_size_[1] + pad_size[0]
        ]  # 2D
    else:
        # Decide on number of histogram samples
        num_hist_samples = num_loss_passes
        overall_stochastic_logits_hist = torch.empty((1, 4, 181, 217, 181, num_hist_samples))
        overall_true_seg_net_out_hist = torch.empty((1, 4, 181, 217, 181, num_hist_samples))

        output_rois = list()
        unc_output_rois = list()
        # Have uncertainty, therefore have MANY outputs, but only have ONE pass through network
        for data in slice_batches:
            if inputs_type == tuple:
                seg_prob, unc_prob, _ = predictor(data, phys_inputs)  # batched patch segmentation
                output_rois.append(seg_prob)
                print(f'The mean of the segmentation logits is {torch.mean(seg_prob)}')
                print(f'The mean of the uncertainty logits is {torch.mean(unc_prob)}')
                if num_loss_passes > 20:
                    unc_prob *= 10**9
                    unc_prob[unc_prob > 10] = 5
                unc_output_rois.append(unc_prob)
            elif inputs_type != tuple:
                seg_prob, unc_prob, _ = predictor(data)  # batched patch segmentation
                output_rois.append(seg_prob)
                print(f'The mean of the segmentation logits is {torch.mean(seg_prob)}')
                print(f'The mean of the uncertainty logits is {torch.mean(unc_prob)}')
                if num_loss_passes > 20:
                    unc_prob *= 10**9
                    unc_prob[unc_prob > 10] = 5
                unc_output_rois.append(unc_prob)
        # Get shape of logits
        logits_shape = list(seg_prob.shape)
        # Now want an array of randomly normally distributed samples size of logits x num samples
        # logits_shape.append(num_hist_samples)
        inf_ax = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda:0")),
                                            torch.tensor(1.0).to(device=torch.device("cuda:0")))
        # inf_noise_array = torch.empty(logits_shape).normal_(mean=0, std=1)
        # Loop through samples

        for infpass in range(num_hist_samples):
            true_output_rois = list()
            true_unc_output_rois = list()
            # print(f'The lengths of rois are {len(output_rois)}, {len(unc_output_rois)}')
            for roi, unc_roi in zip(output_rois, unc_output_rois):
                # output_rois = list()
                # unc_output_rois = list()m
                # Repeat steps above to get more samples
                # noise_sample = inf_noise_array[..., infpass]
                stochastic_logits = roi + unc_roi * inf_ax.sample(logits_shape)  # noise_sample
                # print(f'The sigma mean is {torch.mean(unc_roi)}, logits mean is {torch.mean(roi)}')
                # print(
                #     f'The logits, sigma, ax sizes are: {roi.shape}, {unc_roi.shape}, {inf_ax.sample(logits_shape).shape}')
                # print(
                #     f'A little ax check: {inf_ax.sample(logits_shape)[0, 0, 0, 0, 0]}, {inf_ax.sample(logits_shape)[0, 1, 0, 0, 0]}')
                true_seg_net_out = torch.softmax(stochastic_logits, dim=1)
                # print(f'The stochastic logits shapes are {stochastic_logits.shape}, {true_seg_net_out.shape}')
                true_output_rois.append(true_seg_net_out)
                true_unc_output_rois.append(stochastic_logits)

            # stitching output image
            # print(f'The true output rois tensor shapes are {true_output_rois[0].shape}')
            output_classes = true_output_rois[0].shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)

            # Create importance map
            importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode,
                                                    device=inputs.device)

            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

            # slic_index, zero to len(slices) in increments of sw_batch_size
            for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
                slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

                # store the result in the proper location of the full output. Apply weights from importance map.
                for curr_index in slice_index_range:
                    curr_slice = slices[curr_index]
                    if len(curr_slice) == 3:
                        # print(output_image.shape, curr_slice, importance_map.shape, true_output_rois[window_id].shape)
                        output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                                importance_map * true_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                    else:
                        output_image[0, :, curr_slice[0], curr_slice[1]] += (
                                importance_map * true_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

            # account for any overlapping sections
            output_image /= count_map

            if num_spatial_dims == 3:
                output_image = output_image[
                       ...,
                       pad_size[4]: image_size_[0] + pad_size[4],
                       pad_size[2]: image_size_[1] + pad_size[2],
                       pad_size[0]: image_size_[2] + pad_size[0],
                       ]
                overall_true_seg_net_out_hist[..., infpass] = output_image
            else:
                output_image = output_image[
                       ..., pad_size[2]: image_size_[0] + pad_size[2], pad_size[0]: image_size_[1] + pad_size[0]
                       ]  # 2D
                overall_true_seg_net_out_hist[..., infpass] = output_image

            # Uncertainty part
            # stitching output image
            output_classes = true_unc_output_rois[0].shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)

            # Create importance map
            importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode,
                                                    device=inputs.device)

            # allocate memory to store the full output and the count for overlapping parts
            unc_output = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

            for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
                slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

                # store the result in the proper location of the full output. Apply weights from importance map.
                for curr_index in slice_index_range:
                    curr_slice = slices[curr_index]
                    if len(curr_slice) == 3:
                        unc_output[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                                importance_map * true_unc_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                    else:
                        unc_output[0, :, curr_slice[0], curr_slice[1]] += (
                                importance_map * true_unc_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

            # account for any overlapping sections
            unc_output /= count_map

            if num_spatial_dims == 3:
                unc_output = unc_output[
                               ...,
                               pad_size[4]: image_size_[0] + pad_size[4],
                               pad_size[2]: image_size_[1] + pad_size[2],
                               pad_size[0]: image_size_[2] + pad_size[0],
                               ]
                overall_stochastic_logits_hist[..., infpass] = unc_output
            else:
                unc_output = unc_output[
                               ..., pad_size[2]: image_size_[0] + pad_size[2],
                               pad_size[0]: image_size_[1] + pad_size[0]
                               ]  # 2D
                overall_stochastic_logits_hist[..., infpass] = unc_output

        # Sigma part
        # stitching output image
        output_classes = unc_output_rois[0].shape[1]
        output_shape = [batch_size, output_classes] + list(image_size)

        # Create importance map
        importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode,
                                                device=inputs.device)

        # allocate memory to store the full output and the count for overlapping parts
        sigma_output = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
        count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
        for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
            slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

            # store the result in the proper location of the full output. Apply weights from importance map.
            for curr_index in slice_index_range:
                curr_slice = slices[curr_index]
                if len(curr_slice) == 3:
                    sigma_output[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                            importance_map * unc_output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                else:
                    sigma_output[0, :, curr_slice[0], curr_slice[1]] += (
                            importance_map * unc_output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

        # account for any overlapping sections
        sigma_output /= count_map

        if num_spatial_dims == 3:
            sigma_output = sigma_output[
                           ...,
                           pad_size[4]: image_size_[0] + pad_size[4],
                           pad_size[2]: image_size_[1] + pad_size[2],
                           pad_size[0]: image_size_[2] + pad_size[0],
                           ]
        else:
            sigma_output = sigma_output[
                           ..., pad_size[2]: image_size_[0] + pad_size[2],
                           pad_size[0]: image_size_[1] + pad_size[0]
                           ]  # 2D
        return overall_true_seg_net_out_hist, overall_stochastic_logits_hist, sigma_output


def _get_scan_interval(image_size, roi_size, num_spatial_dims: int, overlap: float):
    assert len(image_size) == num_spatial_dims, "image coord different from spatial dims."
    assert len(roi_size) == num_spatial_dims, "roi coord different from spatial dims."

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            # scan interval is (1-overlap)*roi_size
            scan_interval.append(int(roi_size[i] * (1 - overlap)))
    return tuple(scan_interval)


def kernel_size_calculator(patch_size):
    from math import floor
    if patch_size > 160:
        win_size = 11
    else:
        win_size = floor(((patch_size / 2 ** 4) + 1) / 2)

        if win_size <= 1:
            raise ValueError(
                "Window size for MS-SSIM can't be calculated. Please increase patch_size's smallest dimension."
            )
        # Window size must be odd
        if win_size % 2 == 0:
            win_size += 1
    return win_size


def basename_extractor(full_file_path, keep_extension=True):
    extracted_filename = os.path.basename(full_file_path)
    if not keep_extension:
        extracted_filename = extracted_filename.split('.')[0]
    return extracted_filename
