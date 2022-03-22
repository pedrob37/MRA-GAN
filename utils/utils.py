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

    # Save models to the disk
    def save_networks(self, which_epoch, current_iter, current_fold, models_dir, opt):
        # Define ONE file for saving ALL state dicts
        G_A.cpu()
        G_B.cpu()
        D_A.cpu()
        D_B.cpu()
        save_filename = f'epoch_{which_epoch}_checkpoint_iters_{current_iter}_fold_{current_fold}.pth'
        current_state_dict = {
                              'G_A_optimizer_state_dict': G_A_optimizer.state_dict(),
                              'D_A_optimizer_state_dict': D_A_optimizer.state_dict(),
                              'G_B_optimizer_state_dict': G_B_optimizer.state_dict(),
                              'D_B_optimizer_state_dict': D_B_optimizer.state_dict(),
                              'epoch': which_epoch,
                              'running_iter': current_iter,
                              'batch_size': opt.batch_size,
                              'patch_size': opt.patch_size,
                              'G_A_state_dict': G_A.state_dict(),
                              'G_B_state_dict': G_B.state_dict(),
                              'D_A_state_dict': D_A.state_dict(),
                              'D_B_state_dict': D_B.state_dict(),
                              }
        G_A.cuda()
        G_B.cuda()
        D_A.cuda()
        D_B.cuda()
        for name in self.model_names:
            if isinstance(name, str):
                save_path = os.path.join(models_dir, save_filename)
                net = getattr(self, 'net' + name)
                net.cpu()
                if torch.cuda.is_available():
                    current_state_dict[f'net_{name}_state_dict'] = net.state_dict()
                    net.cuda()
        # Save aggregated checkpoint file
        torch.save(current_state_dict, save_path)

    def write_logs(self, training=True, step=None, current_writer=None):
        losses = self.get_current_losses()
        if training:
            current_writer.add_scalars('Loss/Adversarial',
                                       {"Generator_A": losses["G_A"],
                                        "Generator_B": losses["G_B"],
                                        "Discriminator_A": losses["D_A"],
                                        "Discriminator_B": losses["D_B"]}, step)
        else:
            current_writer.add_scalars('Loss/Val_Adversarial',
                                       {"Generator_A": losses["G_A"],
                                        "Generator_B": losses["G_B"],
                                        "Discriminator_A": losses["D_A"],
                                        "Discriminator_B": losses["D_B"]}, step)

    @ staticmethod
    def normalise_images(array):
        import numpy as np
        return (array - np.min(array)) / (np.max(array) - np.min(array))

    def write_images(self, training=True, step=None, current_writer=None, current_opt=None, current_fold=None):
        # Shape checks
        # print("Basic input and output shape checks!")
        # print(self.real_B.shape,
        #       self.real_A.shape,
        #       self.fake_B.shape,
        #       self.fake_A.shape,
        #       self.rec_B.shape,
        #       self.rec_A.shape,
        #       )
        # self.fake_B = self.netG_A(self.real_A.to(device))
        # self.fake_A = self.netG_B(self.real_B.to(device))
        # if not self.opt.coordconv:
        #     self.rec_A = self.netG_B(self.fake_B.to(device))
        #     self.rec_B = self.netG_A(self.fake_A.to(device))
        # else:
        #     self.rec_A = self.netG_B(torch.cat((self.fake_B.to(device), self.coords), dim=1))
        #     self.rec_B = self.netG_A(torch.cat((self.fake_A.to(device), self.coords), dim=1))

        if training:
            # Reals
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.real_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Visuals/Real_B_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.real_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Visuals/Real_A_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)

            # Generated
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.fake_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Visuals/Fake_B_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.fake_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Visuals/Fake_A_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.rec_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Visuals/Rec_B_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.rec_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Visuals/Rec_A_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
        else:
            # Reals
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.real_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Validation/Real_B_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.real_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Validation/Real_A_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)

            # Generated
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.fake_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Validation/Fake_B_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.fake_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Validation/Fake_A_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.rec_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Validation/Rec_B_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)
            img2tensorboard.add_animated_gif(writer=current_writer,
                                             image_tensor=self.normalise_images(self.rec_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                             tag=f'Validation/Rec_A_fold_{current_fold}',
                                             max_out=current_opt.patch_size // 4,
                                             scale_factor=255, global_step=step)

        # import nibabel as nib
        # import os
        #
        # def save_img(image, affine, filename):
        #     nifti_img = nib.Nifti1Image(image, affine)
        #     if os.path.exists(filename):
        #         raise OSError("File already exists! Killing job")
        #     else:
        #         nib.save(nifti_img, filename)
        #
        # import numpy as np
        # rand_num = np.random.randint(10000)
        # save_img(self.normalise_images(self.fake_A[0, 0, ...].cpu().detach().numpy()), None, f"/nfs/home/pedro/Outputs-MRA-GAN/fake_A_{rand_num}.nii.gz")
        # save_img(self.normalise_images(self.fake_B[0, 0, ...].cpu().detach().numpy()), None, f"/nfs/home/pedro/Outputs-MRA-GAN/fake_B_{rand_num}.nii.gz")


""" Variational Autoencoder

Based on
https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI/blob/master/models/variational_autoencoder.py
"""
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, z_dim=512, dropout_rate=0.0):
        super().__init__()
        self.z_dim = z_dim

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
            nn.Conv3d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
        )

        self.intermediate_conv = nn.Conv3d(256, 16, 1)
        self.mu_layer = nn.Linear(2800, z_dim)
        self.sigma_layer = nn.Linear(2800, z_dim)
        self.dropout_mu_layer_enc = nn.Dropout3d(p=dropout_rate)
        self.dropout_sigma_layer_enc = nn.Dropout3d(p=dropout_rate)

        self.dec_dense = nn.Linear(z_dim, 2800)
        self.dropout_layer_dec = nn.Dropout3d(p=dropout_rate)
        self.intermediate_conv_reverse = nn.Conv3d(16, 128, 1)

        self.decoder = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 1, 1),
        )

    def encode(self, x):
        temp_out = self.encoder(x)

        temp_out = self.intermediate_conv(temp_out)
        flatten = temp_out.view(-1, 2800)

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

    def sampling(self, z_mu, z_sigma):
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def decode(self, z_vae):
        reshaped = self.dropout_layer_dec(self.dec_dense(z_vae)).view(-1, 16, 5, 7, 5)
        temp_out = self.intermediate_conv_reverse(reshaped)
        temp_out = self.decoder(temp_out)

        return temp_out

    def forward(self, x):
        # z_mu, z_sigma = self.encode(x)
        # z = self.sampling(z_mu, z_sigma)
        # reconstruction = self.decode(z)
        z = self.encode(x)
        return z

    def reconstruct(self, x):
        z_mu, _ = self.encode(x)
        reconstruction = self.decode(z_mu)
        return reconstruction