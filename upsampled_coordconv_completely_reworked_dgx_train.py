import os
import random

import torch
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
import time
from models import create_model
from utils.visualizer import Visualizer
from utils.utils import create_path, save_img, CoordConvd, kernel_size_calculator, create_folds, ClipRanged
import monai
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from VSeg.scripts.reworked_preprocessing import preprocessing
import torch.nn as nn
from models.pix2pix_disc_networks import NoisyMultiscaleDiscriminator3D, GANLoss
from monai.networks.layers.factories import Norm
import monai.visualize.img2tensorboard as img2tensorboard
from itertools import cycle
from monai.transforms import (Compose,
                              LoadImaged,
                              AddChanneld,
                              RandAffined,
                              RandBiasFieldd,
                              ToTensord,
                              RandSpatialCropSamplesd,
                              RandWeightedCropd,
                              SpatialCropd,
                              NormalizeIntensityd,
                              RandGaussianSmoothd,
                              RandGaussianNoiseD,
                              RandGaussianNoise,
                              )
from utils.MSSSIM.pytorch_msssim.ssim import SSIM
# from models.UNet_adnless import UNet
# from models.unet import UNet
# from monai.networks.nets import UNet
from chinai.networks.nets import UNet
from chinai.networks.layers import Norm


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    from monai.utils import set_determinism

    random_seed_number = np.random.randint(0, 100, 1)[0]
    set_determinism(random_seed_number)
    print(f"The random seed is {random_seed_number}")

    # GPU count
    print(f"There are {torch.cuda.device_count()} GPUs available!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model choice: Trilinear, nearest neighbour, or nearest neighbour + subpixel convolution
    if opt.upsampling_method == 'nearest':
        from models.model_nn_upsamps_norm import nnUNet

        print("Using nearest neighbour interpolation for upsampling!")
    elif opt.upsampling_method == 'exp-nearest':
        from models.extended_model_nn_upsamps_norm import nnUNet

        print("Using Expanded nearest neighbour interpolation for upsampling!")
    elif opt.upsampling_method == 'trilinear':
        from models.model_upsamps_norm import nnUNet

        print("Using trilinear interpolation for upsampling!")
    elif opt.upsampling_method == 'subpixel':
        from models.model_subpixel_upsamps_norm import nnUNet

        print("Using nearest neighbour interpolation + subpixel convolution for upsampling!")
    elif opt.upsampling_method == "trans":
        from models.model_transconvs_norm import nnUNet

        print("Using transposed convolutions for upsampling!")
    elif opt.upsampling_method == "even-trans":
        from models.model_even_transconvs_norm import nnUNet

        print("Using even transposed convolutions for upsampling!")
    elif opt.upsampling_method == "exp-trans":
        from models.extended_model_transconvs_norm import nnUNet

        print("Using Expanded transposed convolutions for upsampling!")
    else:
        raise SyntaxError("Missing upsampling method parameter!")

    ### MONAI
    # Data directory
    images_dir = os.path.join(opt.data_path, 'Images')  # MRA!
    labels_dir = os.path.join(opt.data_path, 'Labels')  # Binary!
    if opt.t1_aid:
        t1s_dir = os.path.join(opt.data_path, 'T1s')  # T1s!

    # Read csv + add directory to filenames
    if opt.use_csv:
        df = pd.read_csv(opt.csv_file)
        df['Label_Filename'] = df['Filename']
        if opt.t1_aid:
            df['T1'] = df['Filename']
            df['T1'] = t1s_dir + '/' + df['Filename'].astype(str)
            # Shuffling
            df['T1'] = df['T1'].sample(frac=1).values
        df['Filename'] = images_dir + '/' + df['Filename'].astype(str)
        df['Label_Filename'] = labels_dir + '/' + df['Label_Filename'].astype(str)
        # Shuffling
        df['Label_Filename'] = df['Label_Filename'].sample(frac=1).values
        num_folds = df.fold.nunique()

        # Inference fold assignment
        inf_fold = 9
        inf_df = df[df.fold == inf_fold]
        inf_df.reset_index(drop=True, inplace=True)
    else:
        num_folds = 1

    # Check real label
    print(f"The target real label is {opt.real_label}")

    # Losses
    # Define discriminator loss and other related variables
    real_label = opt.real_label
    fake_label = 0

    # Pix2PixHD loss setup
    multiscale_discriminator_loss = GANLoss(use_lsgan=True,
                                            target_real_label=real_label,
                                            target_fake_label=fake_label,
                                            tensor=torch.cuda.FloatTensor
                                            )

    # Other cycle/ idt losses
    criterionCycleA = torch.nn.L1Loss()
    if opt.gen_dice_cycle:
        criterionCycleB = monai.losses.generalized_dice()
    else:
        criterionCycleB = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()
    criterionDice = monai.losses.generalized_dice()

    # MS-SSIM
    if opt.msssim:
        gaussian_kernel_size = kernel_size_calculator(opt.patch_size)
        # criterionMSSSIM = SSIM(data_range=45,
        #                        gaussian_kernel_size=gaussian_kernel_size,
        #                        gradient_based=True,
        #                        star_based=False,
        #                        gradient_masks_weights=None,
        #                        input_range_correction="zero_bounded")
        # criterionMSSSIM = SSIM(data_range=45,
        #                        gaussian_kernel_size=gaussian_kernel_size,
        #                        gradient_based=True,
        #                        star_based=False,
        #                        gradient_masks_weights=None,
        #                        input_range_correction="zero_bounded")
        if opt.standard_msssim and not opt.znorm:
            criterionMSSSIM = SSIM(data_range=1.0,
                                   gaussian_kernel_size=gaussian_kernel_size,
                                   gradient_based=True,
                                   star_based=False,
                                   gradient_masks_weights=None,
                                   input_range_correction=opt.range_correction)
        elif opt.standard_msssim and opt.znorm:
            criterionMSSSIM = SSIM(data_range=opt.data_range,  # z-norm images have higher max!
                                   gaussian_kernel_size=gaussian_kernel_size,
                                   gradient_based=True,
                                   star_based=False,
                                   gradient_masks_weights=None,
                                   input_range_correction=opt.range_correction)
        elif not opt.standard_msssim and not opt.znorm:
            criterionMSSSIM = SSIM(data_range=1.0,
                                   gaussian_kernel_size=gaussian_kernel_size,
                                   gradient_based=True,
                                   star_based=False,
                                   gradient_masks_weights=None,
                                   multi_scale_weights=(0.8, 0.1, 0.05, 0.025, 0.025),
                                   input_range_correction=opt.range_correction)
        elif not opt.standard_msssim and opt.znorm:
            criterionMSSSIM = SSIM(data_range=opt.data_range,
                                   gaussian_kernel_size=gaussian_kernel_size,
                                   gradient_based=True,
                                   star_based=False,
                                   gradient_masks_weights=None,
                                   multi_scale_weights=(0.8, 0.1, 0.05, 0.025, 0.025),
                                   input_range_correction=opt.range_correction)


    def normalise_images(array):
        import numpy as np
        return (array - np.min(array)) / (np.max(array) - np.min(array))


    def discriminator_loss(gen_images, real_images, discriminator, real_label_flip_chance=0.25):
        """
        The discriminator loss is calculated by comparing Discriminator
        prediction for real and generated images.
        """

        real_disc_prediction = discriminator.forward(real_images)
        fake_disc_prediction = discriminator.forward(gen_images.detach())

        # Calculate losses across all discriminator levels
        flip_labels = np.random.uniform(0, 1)
        if flip_labels < real_label_flip_chance:
            realloss = multiscale_discriminator_loss(real_disc_prediction, target_is_real=False)
        else:
            realloss = multiscale_discriminator_loss(real_disc_prediction, target_is_real=True)
        genloss = multiscale_discriminator_loss(fake_disc_prediction, target_is_real=False)

        # Calculate accuracies for every discriminator
        real_disc_accuracy = torch.mean(real_disc_prediction[0][0].float())
        fake_disc_accuracy = 1 - torch.mean(fake_disc_prediction[0][0].float())

        # real_disc_sum_ds1 = real_disc_prediction[0][0].float().sum(axis=(1, 2, 3, 4)) / real_disc_prediction[0][0][
        #     0, ...].nelement()
        # fake_disc_sum_ds1 = fake_disc_prediction[0][0].float().sum(axis=(1, 2, 3, 4)) / fake_disc_prediction[0][0][
        #     0, ...].nelement()

        # print("Real, Fake Accs:", torch.mean(real_disc_prediction[0][0].float()), torch.mean(fake_disc_prediction[0][0].float()))

        # real_disc_accuracy = ((real_disc_sum_ds1 > 0.5) == round(real_label)).float().sum() / real_images.shape[0]
        # fake_disc_accuracy = ((fake_disc_sum_ds1 > 0.5) == fake_label).float().sum() / real_images.shape[0]
        # print("Real, Fake Sums:", real_disc_sum_ds1.detach().item(), fake_disc_sum_ds1.detach().item())
        # print("Real, Fake Accs:", real_disc_accuracy_ds1.detach().item(), fake_disc_accuracy_ds1.detach().item())
        # if opt.n_D == 2:
        #     real_disc_sum_ds2 = real_disc_prediction[1][0].float().sum(axis=(1, 2, 3, 4)) / real_disc_prediction[1][0][
        #         0, ...].nelement()
        #     fake_disc_sum_ds2 = fake_disc_prediction[1][0].float().sum(axis=(1, 2, 3, 4)) / fake_disc_prediction[1][0][
        #         0, ...].nelement()
        #     real_disc_accuracy_ds2 = ((real_disc_sum_ds2 > 0.5) == real_label).float().sum() / real_images.shape[0]
        #     fake_disc_accuracy_ds2 = ((fake_disc_sum_ds2 > 0.5) == fake_label).float().sum() / real_images.shape[0]
        #
        #     real_disc_accuracy = (real_disc_accuracy_ds1 + real_disc_accuracy_ds2) / 2
        #     fake_disc_accuracy = (fake_disc_accuracy_ds1 + fake_disc_accuracy_ds2) / 2
        # elif opt.n_D == 1:
        #     real_disc_accuracy = real_disc_accuracy_ds1
        #     fake_disc_accuracy = fake_disc_accuracy_ds1
        # print(real_disc_accuracy.detach().cpu().item())

        return (genloss + realloss) / 2, \
               real_disc_accuracy.cpu().detach().item(), \
               fake_disc_accuracy.cpu().detach().item(), \
               (real_disc_accuracy.cpu().detach().item() + fake_disc_accuracy.cpu().detach().item()) / 2, \
               real_disc_prediction, fake_disc_prediction


    def perceptual_loss(real_images, rec_images, network_choice, num_slices):
        """
        Perceptual loss between original image and reconstructed image
        """
        # Sagittal
        x_2d = rec_images.float().permute(0, 2, 1, 3, 4).contiguous().view(-1,
                                                                           rec_images.shape[1],
                                                                           rec_images.shape[3],
                                                                           rec_images.shape[4])
        y_2d = real_images.float().permute(0, 2, 1, 3, 4).contiguous().view(-1,
                                                                            real_images.shape[1],
                                                                            real_images.shape[3],
                                                                            real_images.shape[4])
        indices = torch.randperm(x_2d.size(0))[:num_slices]
        selected_x_2d = x_2d[indices]
        selected_y_2d = y_2d[indices]

        p_loss_sagital = torch.mean(
            network_choice.forward(
                selected_x_2d.float(),
                selected_y_2d.float()
            )
        )

        # Axial
        x_2d = rec_images.float().permute(0, 4, 1, 2, 3).contiguous().view(-1,
                                                                           rec_images.shape[1],
                                                                           rec_images.shape[2],
                                                                           rec_images.shape[3])
        y_2d = real_images.float().permute(0, 4, 1, 2, 3).contiguous().view(-1,
                                                                            real_images.shape[1],
                                                                            real_images.shape[2],
                                                                            real_images.shape[3])
        indices = torch.randperm(x_2d.size(0))[:num_slices]
        selected_x_2d = x_2d[indices]
        selected_y_2d = y_2d[indices]

        p_loss_axial = torch.mean(
            network_choice.forward(
                selected_x_2d.float(),
                selected_y_2d.float()
            )
        )

        # Coronal
        x_2d = rec_images.float().permute(0, 3, 1, 2, 4).contiguous().view(-1,
                                                                           rec_images.shape[1],
                                                                           rec_images.shape[2],
                                                                           rec_images.shape[4])
        y_2d = real_images.float().permute(0, 3, 1, 2, 4).contiguous().view(-1,
                                                                            real_images.shape[1],
                                                                            real_images.shape[2],
                                                                            real_images.shape[4])
        indices = torch.randperm(x_2d.size(0))[:num_slices]
        selected_x_2d = x_2d[indices]
        selected_y_2d = y_2d[indices]

        p_loss_coronal = torch.mean(
            network_choice.forward(
                selected_x_2d.float(),
                selected_y_2d.float()
            )
        )

        p_loss = p_loss_sagital + p_loss_axial + p_loss_coronal

        return p_loss


    def generator_loss(gen_images, discriminator):
        """
        The generator loss is calculated by determining how realistic
        the discriminator classifies the generated images.
        """
        output = discriminator.forward(gen_images)
        gen_fake_loss = multiscale_discriminator_loss(output, target_is_real=True)

        # Calculate accuracies for every discriminator
        # gen_sum_ds1 = output[0][0].float().sum(axis=(1, 2, 3, 4)) / output[0][0][0, ...].nelement()
        # gen_acc_ds1 = ((gen_sum_ds1 > 0.5) == round(real_label)).float().sum() / gen_images.shape[0]

        gen_acc_ds1 = 1 - torch.mean(output[0][0].float())

        return gen_fake_loss, gen_acc_ds1.cpu().detach().item()


    # Augmentations/ Transforms
    # if opt.cycle_noise:
    post_gen_noise = RandGaussianNoise(prob=1.0, mean=0.0, std=opt.gen_noise_std)
    if opt.t1_aid:
        general_keys_list = ['image', 'label', 'T1']
        crop_keys_list = ['image', 'label', 'coords', 'T1']
    else:
        general_keys_list = ['image', 'label']
        crop_keys_list = ['image', 'label', 'coords']
    if opt.phase == "train":
        if opt.augmentation_level == "heavy":
            train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                        RandAffined(keys=["image", "label", "coords"],
                                                    scale_range=(0.1, 0.1, 0.1),
                                                    rotate_range=(0.25, 0.25, 0.25),
                                                    translate_range=(20, 20, 20),
                                                    mode=("bilinear", "nearest", "nearest"),
                                                    as_tensor_output=False, prob=1.0,
                                                    padding_mode=('zeros', 'zeros', 'border')),
                                        RandGaussianSmoothd(keys=["image"], prob=0.25,  # 0.2
                                                            sigma_x=(0.25, 0.3),
                                                            sigma_y=(0.25, 0.3),
                                                            sigma_z=(0.25, 0.3)),
                                        RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.1, 0.25),
                                                       prob=0.25),  # Odd behaviour...
                                        NormalizeIntensityd(keys=['image'], channel_wise=True),
                                        RandGaussianNoiseD(keys=["image"], std=0.3, prob=0.5),
                                        RandSpatialCropSamplesd(keys=["image", "label", "coords"],
                                                                roi_size=(
                                                                    opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label', 'coords'])])
        elif opt.augmentation_level == "light":
            if opt.t1_aid:
                train_transform_list = [LoadImaged(keys=['image', 'label', 'T1']),
                                        AddChanneld(keys=['image', 'label', 'T1']),
                                        # monai.transforms.utils_pytorch_numpy_unification.clip(a, a_min, a_max),
                                        ClipRanged(keys=["T1"], b_min=-110, b_max=6000),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                        RandAffined(keys=["image", "label", "coords", 'T1'],
                                                    scale_range=(0.1, 0.1, 0.1),
                                                    rotate_range=(0.25, 0.25, 0.25),
                                                    translate_range=(20, 20, 20),
                                                    mode=("bilinear", "nearest", "nearest", 'bilinear'),
                                                    as_tensor_output=False, prob=1.0,
                                                    padding_mode=('zeros', 'zeros', 'border', 'zeros'))]
                if opt.znorm:
                    train_transform_list.append(NormalizeIntensityd(keys=['image', 'T1'], channel_wise=True))
            else:
                train_transform_list = [LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
                if opt.weighted_sampling == "cropped":
                    train_transform_list.append(SpatialCropd(keys=['image', 'label', 'coords'],
                                                             roi_size=(192, 224, 128),
                                                             roi_center=(0, 0, 0)))

                train_transform_list.append(RandAffined(keys=["image", "label", "coords"],
                                                        scale_range=(0.1, 0.1, 0.1),
                                                        rotate_range=(0.25, 0.25, 0.25),
                                                        translate_range=(20, 20, 20),
                                                        mode=("bilinear", "nearest", "nearest"),
                                                        as_tensor_output=False, prob=1.0,
                                                        padding_mode=('zeros', 'zeros', 'border')))
                if opt.znorm:
                    train_transform_list.append(NormalizeIntensityd(keys=['image'], channel_wise=True))
        elif opt.augmentation_level == "none":
            if opt.t1_aid:
                train_transform_list = [LoadImaged(keys=['image', 'label', 'T1']),
                                        AddChanneld(keys=['image', 'label', 'T1']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
                if opt.znorm:
                    train_transform_list.append(NormalizeIntensityd(keys=['image', 'T1'], channel_wise=True))
            else:
                train_transform_list = [LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
                if opt.znorm:
                    train_transform_list.append(NormalizeIntensityd(keys=['image'], channel_wise=True))
        # Validation
        if opt.t1_aid:
            val_transform_list = [LoadImaged(keys=['image', 'label', 'T1']),
                                  AddChanneld(keys=['image', 'label', 'T1']),
                                  ClipRanged(keys=["T1"], b_min=-110, b_max=6000),
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
            if opt.znorm:
                val_transform_list.append(NormalizeIntensityd(keys=['image', 'T1'], channel_wise=True))
        else:
            val_transform_list = [LoadImaged(keys=['image', 'label']),
                                  AddChanneld(keys=['image', 'label']),
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
            if opt.weighted_sampling == "cropped":
                val_transform_list.append(SpatialCropd(keys=['image', 'label', 'coords'],
                                                       roi_size=(192, 224, 128),
                                                       roi_center=(0, 0, 0)))
            if opt.znorm:
                val_transform_list.append(NormalizeIntensityd(keys=['image'], channel_wise=True))

        # Extend remaining transforms
        if opt.weighted_sampling == "weighted":
            train_transform_list.extend([RandWeightedCropd(keys=crop_keys_list,
                                                           w_key='image',
                                                           spatial_size=(
                                                               opt.patch_size, opt.patch_size, 128),
                                                           num_samples=1),
                                         ToTensord(keys=crop_keys_list)])
            val_transform_list.extend([RandWeightedCropd(keys=crop_keys_list,
                                                         w_key='image',
                                                         spatial_size=(
                                                             opt.patch_size, opt.patch_size, 128),
                                                         num_samples=1),
                                       ToTensord(keys=crop_keys_list)])
        elif opt.weighted_sampling == "cropped":
            print("Cropping images, so no sampling necessary")
        else:
            train_transform_list.extend([RandSpatialCropSamplesd(keys=crop_keys_list,
                                                                 roi_size=(
                                                                     opt.patch_size, opt.patch_size, opt.patch_size),
                                                                 random_center=True,
                                                                 random_size=False,
                                                                 num_samples=1),
                                         ToTensord(keys=crop_keys_list)])
        val_transform_list.extend([RandSpatialCropSamplesd(keys=crop_keys_list,
                                                           roi_size=(
                                                               opt.patch_size, opt.patch_size, opt.patch_size),
                                                           random_center=True,
                                                           random_size=False,
                                                           num_samples=1),
                                   ToTensord(keys=crop_keys_list)])

        # Compose
        train_transforms = Compose(train_transform_list)
        val_transforms = Compose(val_transform_list)

    elif opt.phase == "test":
        # Inference
        from monai.inferers import sliding_window_inference
        from monai.inferers import SimpleInferer

        simple_inferer = SimpleInferer()
        from utils.utils import basename_extractor

        if opt.t1_aid:
            inf_transform_list = [LoadImaged(keys=['image', 'label', 'T1']),
                                  AddChanneld(keys=['image', 'label', 'T1']),
                                  ClipRanged(keys=["T1"], b_min=-110, b_max=6000),
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
            if opt.znorm:
                inf_transform_list.append(NormalizeIntensityd(keys=['image', 'T1'], channel_wise=True))
        else:
            inf_transform_list = [LoadImaged(keys=['image', 'label']),
                                  AddChanneld(keys=['image', 'label']),
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
            if opt.weighted_sampling == "cropped":
                inf_transform_list.append(SpatialCropd(keys=['image', 'label', 'coords'],
                                                       roi_size=(192, 224, 128),
                                                       roi_center=(0, 0, 0)))
            if opt.znorm:
                inf_transform_list.append(NormalizeIntensityd(keys=['image'], channel_wise=True))

        # Extend
        if opt.t1_aid:
            inf_transform_list.extend([ToTensord(keys=['image', 'label', 'coords', 'T1'])])
        else:
            inf_transform_list.extend([ToTensord(keys=['image', 'label', 'coords'])])

        # Compose
        inf_transforms = Compose(inf_transform_list)

    ## Relevant job directories
    base_dir = opt.base_dir
    if opt.phase == "train":
        # Main directories
        CACHE_DIR = f"{base_dir}/Outputs-MRA-GAN/Cache/{opt.job_name}"
        FIG_DIR = f"{base_dir}/Outputs-MRA-GAN/Figures/{opt.job_name}"
        LOG_DIR = f'{base_dir}/Outputs-MRA-GAN/Logs/{opt.job_name}'
        SLIM_LOG_DIR = f'{base_dir}/Outputs-MRA-GAN/Slim_Logs/{opt.job_name}'
        MODELS_DIR = f"{base_dir}/Outputs-MRA-GAN/Models/{opt.job_name}"

        # Create directories
        create_path(CACHE_DIR)
        create_path(FIG_DIR)
        create_path(LOG_DIR)
        create_path(SLIM_LOG_DIR)
        create_path(MODELS_DIR)
    elif opt.phase == "test" and opt.job_name.startswith("inf"):
        opt.job_name = opt.job_name.split('-', 1)[1]

        # Directories should already exist
        CACHE_DIR = f"{base_dir}/Outputs-MRA-GAN/Cache/{opt.job_name}"
        FIG_DIR = f"{base_dir}/Outputs-MRA-GAN/Figures/{opt.job_name}"
        LOG_DIR = f'{base_dir}/Outputs-MRA-GAN/Logs/{opt.job_name}'
        SLIM_LOG_DIR = f'{base_dir}/Outputs-MRA-GAN/Slim_Logs/{opt.job_name}'
        MODELS_DIR = f"{base_dir}/Outputs-MRA-GAN/Models/{opt.job_name}"
    else:
        raise NameError("Job phase is test but job name does not start with inf!")

    # Re-assign checkpoints directory
    opt.checkpoints_dir = MODELS_DIR
    opt.log_dir = LOG_DIR

    # Other variables
    val_gap = 5
    LOAD = True
    D_A_acc = None
    D_B_acc = None

    # Folds
    for fold in range(num_folds):
        # Define models and associated variables
        print('\nFOLD', fold)
        # Generators
        if opt.final_act == "leaky":
            G_A_final_act = torch.nn.LeakyReLU()
        elif opt.final_act == "sigmoid":
            G_A_final_act = torch.nn.Sigmoid()
        G_A = nnUNet(4, 1, dropout_level=0, final_act=G_A_final_act)
        if opt.t1_aid:
            G_B = nnUNet(5, 1, dropout_level=0)
        else:
            G_B = nnUNet(4, 1, dropout_level=0)

        if opt.seg_loss:
            # Forward pass through network
            vseg_model = UNet(dimensions=3, in_channels=2, out_channels=2,
                              channels=(16, 32, 64, 128, 256), strides=(1, 1, 1, 1),
                              num_res_units=2, norm=Norm.BATCH).cuda()

            vseg_model.load_state_dict(torch.load(os.path.join(f"{base_dir}/MRA-GAN/VSeg/PretrainedModels",
                                                               "last_model_Nep2000.pth")), strict=True)

            vseg_model.eval()

        if opt.perceptual:
            import lpips

            perceptual_net = lpips.LPIPS(pretrained=True, net='squeeze')
            # if torch.cuda.device_count() > 1:
            perceptual_net = torch.nn.DataParallel(perceptual_net)

        # Discriminators
        D_A = NoisyMultiscaleDiscriminator3D(1, opt.ndf,
                                             opt.n_layers_D,
                                             nn.InstanceNorm3d, True, opt.n_D, False)

        D_B = NoisyMultiscaleDiscriminator3D(1, opt.ndf,
                                             opt.n_layers_D,
                                             nn.InstanceNorm3d, True, opt.n_D, False)

        # Associated variables
        G_A = nn.DataParallel(G_A)
        G_B = nn.DataParallel(G_B)
        D_A = nn.DataParallel(D_A)
        D_B = nn.DataParallel(D_B)

        # Parameter counts
        G_A_total_params = sum(p.numel() for p in G_A.parameters())
        G_B_total_params = sum(p.numel() for p in G_B.parameters())
        D_A_total_params = sum(p.numel() for p in D_A.parameters())
        D_B_total_params = sum(p.numel() for p in D_B.parameters())

        print(f"G_A: {G_A_total_params}\n"
              f"G_B: {G_B_total_params}\n"
              f"D_A: {D_A_total_params}\n"
              f"D_B: {D_B_total_params}\n")

        # Optimizers + schedulers
        import itertools

        G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()),
                                       lr=opt.gen_lr, betas=(0.5, 0.999))
        D_optimizer = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                                       lr=opt.disc_lr, betas=(0.5, 0.999))

        # Domain A
        G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, 0.999)
        D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, 0.999)

        # Loading
        # Model loading
        file_list = os.listdir(path=MODELS_DIR)
        num_files = len(file_list)
        print(f'The number of files is {num_files}')

        if LOAD and num_files > 0 and opt.phase != 'inference':
            # Find latest model
            import glob

            # total_steps = model.load_networks('latest', models_dir=MODELS_DIR, phase=opt.phase)
            model_files = glob.glob(os.path.join(MODELS_DIR, '*.pth'))
            for some_model_file in model_files:
                print(some_model_file)
            sorted_model_files = sorted(model_files, key=os.path.getmtime)
            # Allows inference to be run on nth latest file!
            latest_model_file = sorted_model_files[-1]
            checkpoint = torch.load(latest_model_file, map_location=torch.device(device))
            print(f'Loading {latest_model_file}!')
            loaded_epoch = checkpoint['epoch']
            running_iter = checkpoint['running_iter']
            total_steps = checkpoint['total_steps']

            # Get model file specific to fold
            # loaded_model_file = f'epoch_{loaded_epoch}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
            # checkpoint = torch.load(os.path.join(MODELS_DIR, loaded_model_file), map_location=torch.device(device))

            # Main model variables
            G_A.load_state_dict(checkpoint['G_A_state_dict'])
            G_B.load_state_dict(checkpoint['G_B_state_dict'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            G_scheduler.load_state_dict(checkpoint["G_scheduler_state_dict"])
            # gen_scaler.load_state_dict(checkpoint['scaler'])
            for state in G_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            # Load discriminator-specific variables
            D_A.load_state_dict(checkpoint['D_A_state_dict'])
            D_B.load_state_dict(checkpoint['D_B_state_dict'])
            D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
            D_scheduler.load_state_dict(checkpoint["D_scheduler_state_dict"])
            # gen_scaler.load_state_dict(checkpoint['scaler'])
            for state in D_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            # Ensure that no more loading is done for future folds
            LOAD = False
        elif LOAD and num_files > 0 and opt.phase == 'inference':
            # Find latest model
            import glob

            # total_steps = model.load_networks('latest', models_dir=MODELS_DIR, phase=opt.phase)
            model_files = glob.glob(os.path.join(MODELS_DIR, '*.pth'))
            for some_model_file in model_files:
                print(some_model_file)
            sorted_model_files = sorted(model_files, key=os.path.getmtime)
            # Allows inference to be run on nth latest file!
            latest_model_file = sorted_model_files[-1]
            checkpoint = torch.load(latest_model_file, map_location=torch.device(device))
            print(f'Loading {latest_model_file}!')
            loaded_epoch = checkpoint['epoch']
            running_iter = checkpoint['running_iter']
            total_steps = checkpoint['total_steps']

            # best_model_file = f'epoch_{loaded_epoch}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
            print(f'Loading checkpoint for model: {os.path.basename(latest_model_file)}')
            # # checkpoint = torch.load(os.path.join(SAVE_PATH, best_model_file), map_location=model.device)
            # checkpoint = torch.load(os.path.join(MODELS_DIR, best_model_file), map_location=torch.device(device))
            # Main model variables
            G_A.load_state_dict(checkpoint['G_A_state_dict'])
            G_B.load_state_dict(checkpoint['G_B_state_dict'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            G_scheduler.load_state_dict(checkpoint["G_scheduler_state_dict"])
            # gen_scaler.load_state_dict(checkpoint['scaler'])
            for state in G_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        elif num_files == 0:
            total_steps = 0
            running_iter = 0
            loaded_epoch = 0

        # Push to device
        G_A.cuda()
        G_B.cuda()
        D_A.cuda()
        D_B.cuda()

        if opt.perceptual:
            perceptual_net = perceptual_net.cuda()

        # Writer
        writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f'fold_{fold}'))
        slim_writer = SummaryWriter(log_dir=os.path.join(SLIM_LOG_DIR, f'fold_{fold}'))

        if opt.use_csv:
            # Train / Val split
            val_fold = fold
            excluded_folds = [val_fold, inf_fold]
            train_df = df[~df.fold.isin(excluded_folds)]
            val_df = df[df.fold == val_fold]
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)

            # Print sizes
            logging_interval = int(len(train_df) / 3)
            print(f'The length of the training is {len(train_df)}')
            print(f'The length of the validation is {len(val_df)}')
            print(f'The length of the inference is {len(inf_df)}')

            # Data dicts
            if opt.t1_aid:
                train_data_dict = [{'image': image_name, 'label': label_name, 'T1': t1_name} for
                                   image_name, label_name, t1_name
                                   in zip(train_df.Filename, train_df.Label_Filename, train_df.T1)]
                val_data_dict = [{'image': image_name, 'label': label_name, 'T1': t1_name} for
                                 image_name, label_name, t1_name
                                 in zip(val_df.Filename, val_df.Label_Filename, val_df.T1)]
                inf_data_dict = [{'image': image_name, 'label': label_name, 'T1': t1_name} for
                                 image_name, label_name, t1_name
                                 in zip(inf_df.Filename, inf_df.Label_Filename, inf_df.T1)]
            else:
                train_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                                   in zip(train_df.Filename, train_df.Label_Filename)]
                val_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                                 in zip(val_df.Filename, val_df.Label_Filename)]
                inf_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                                 in zip(inf_df.Filename, inf_df.Label_Filename)]

            # Basic length checks
            assert len(train_df.Filename) == len(train_df.Label_Filename)
            assert len(val_df.Filename) == len(val_df.Label_Filename)

            # Need to shuffle!
            do_shuffling = True

        elif not opt.use_csv:
            import glob

            full_images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
            full_labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

            # Splits: Random NON-GLOBAL shuffle:
            # https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
            random.Random(1).shuffle(full_images)
            random.Random(1).shuffle(full_labels)

            train_images, val_images, inf_images = create_folds(full_images, train_split=0.8, val_split=0.1)
            train_labels, val_labels, inf_labels = create_folds(full_labels, train_split=0.8, val_split=0.1)

            # Re-shuffle: Don't want data to be subject matched!
            random.Random(2).shuffle(train_images)
            random.Random(3).shuffle(train_labels)

            random.Random(5).shuffle(val_images)
            random.Random(6).shuffle(val_labels)

            random.Random(8).shuffle(inf_images)
            random.Random(9).shuffle(inf_labels)

            if opt.t1_aid:
                full_t1s = sorted(glob.glob(os.path.join(t1s_dir, "*.nii.gz")))
                random.Random(1).shuffle(full_t1s)
                train_t1s, val_t1s, inf_t1s = create_folds(full_t1s, train_split=0.8, val_split=0.1)
                random.Random(4).shuffle(train_t1s)
                random.Random(7).shuffle(val_t1s)
                random.Random(10).shuffle(inf_t1s)

                train_data_dict = [{'image': image_name, 'label': label_name, 'T1': t1_name} for
                                   image_name, label_name, t1_name
                                   in zip(cycle(train_images), train_labels, cycle(train_t1s))]
                val_data_dict = [{'image': image_name, 'label': label_name, 'T1': t1_name} for
                                 image_name, label_name, t1_name
                                 in zip(cycle(val_images), val_labels, cycle(val_t1s))]
                inf_data_dict = [{'image': image_name, 'label': label_name, 'T1': t1_name} for
                                 image_name, label_name, t1_name
                                 in zip(cycle(inf_images), inf_labels, cycle(inf_t1s))]
                print(f"Length of inference images, labels, T1s: {len(inf_images)}, {len(inf_labels)}, {len(inf_t1s)}")
            else:
                train_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                                   in zip(cycle(train_images), train_labels)]
                val_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                                 in zip(cycle(val_images), val_labels)]
                inf_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                                 in zip(cycle(inf_images), inf_labels)]

            # Print sizes
            logging_interval = int(len(train_images) / 2)
            print(f'The length of the training is {len(train_images)}')
            print(f'The length of the validation is {len(val_images)}')
            print(f'The length of the inference is {len(inf_images)}')

            # No need to shuffle!
            do_shuffling = False

        if opt.phase == "train":
            # Training + validation loaders
            # train_ds = monai.data.Dataset(data=train_data_dict,
            #                               transform=train_transforms,
            #                               # cache_dir=CACHE_DIR
            #                               )
            train_ds = monai.data.PersistentDataset(data=train_data_dict,
                                                    transform=train_transforms,
                                                    cache_dir=CACHE_DIR
                                                    )

            train_loader = DataLoader(dataset=train_ds,
                                      batch_size=opt.batch_size,
                                      shuffle=do_shuffling,
                                      num_workers=opt.workers,
                                      )

            # val_ds = monai.data.Dataset(data=val_data_dict,
            #                             transform=val_transforms,
            #                             # cache_dir=CACHE_DIR
            #                             )
            val_ds = monai.data.PersistentDataset(data=val_data_dict,
                                                  transform=val_transforms,
                                                  cache_dir=CACHE_DIR
                                                  )

            val_loader = DataLoader(dataset=val_ds,
                                    batch_size=opt.batch_size,
                                    shuffle=do_shuffling,
                                    num_workers=opt.workers,
                                    )
        elif opt.phase == "test":
            inf_ds = monai.data.Dataset(data=inf_data_dict,
                                        transform=inf_transforms,
                                        )

            inf_loader = DataLoader(dataset=inf_ds,
                                    batch_size=1,
                                    shuffle=do_shuffling,
                                    num_workers=opt.workers,
                                    )

        # Model creation
        # model = create_model(opt)
        # model.setup(opt)
        visualizer = Visualizer(opt)

        # Z distribution
        # z_sampler = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda:0")),
        #                                        torch.tensor(1.0).to(device=torch.device("cuda:0")))

        if opt.phase == "train":
            # Epochs
            for epoch in range(loaded_epoch, opt.niter + opt.niter_decay + 1):
                # Model to train mode after potential eval call when running validation
                G_A.train()
                G_B.train()
                D_A.train()
                D_B.train()

                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0

                # Iterations
                for some_iter, train_sample in enumerate(train_loader):
                    # print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
                    # print(train_D_A, train_G_A, train_D_B, train_G_B)
                    iter_start_time = time.time()
                    # print("LI, total_steps", logging_interval, total_steps)
                    if total_steps % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    visualizer.reset()

                    # Determine whether to train D, G, both, or neither
                    if D_A_acc is None:
                        train_D_A = True
                        train_G_B = True
                    elif opt.disc_threshold_low <= D_A_acc <= opt.disc_threshold_high:
                        train_D_A = True
                        train_G_B = True
                    elif D_A_acc < opt.disc_threshold_low:
                        train_D_A = True
                        train_G_B = False
                    elif D_A_acc > opt.disc_threshold_high:
                        train_D_A = False
                        train_G_B = True
                    else:
                        Warning("Non numeric accuracy.")

                    # Determine whether to train D, G, both, or neither
                    if D_B_acc is None:
                        train_D_B = True
                        train_G_A = True
                    elif opt.disc_threshold_low <= D_B_acc <= opt.disc_threshold_high:
                        train_D_B = True
                        train_G_A = True
                    elif D_B_acc < opt.disc_threshold_low:
                        train_D_B = True
                        train_G_A = False
                    elif D_B_acc > opt.disc_threshold_high:
                        train_D_B = False
                        train_G_A = True
                    else:
                        Warning("Non numeric accuracy.")

                    # Training variables: Global
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size

                    # Iteration-specific data
                    if opt.weighted_sampling == "cropped":
                        real_A = train_sample['image'].cuda()
                        real_B = train_sample['label'].cuda()
                        train_coords = train_sample['coords'].cuda()

                        # Names (Not needed for now)
                        image_name = os.path.basename(train_sample["image_meta_dict"]["filename_or_obj"][0])
                        label_name = os.path.basename(train_sample["label_meta_dict"]["filename_or_obj"][0])
                    else:
                        real_A = train_sample[0]['image'].cuda()
                        real_B = train_sample[0]['label'].cuda()
                        train_coords = train_sample[0]['coords'].cuda()

                        # Names (Not needed for now)
                        image_name = os.path.basename(train_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                        label_name = os.path.basename(train_sample[0]["label_meta_dict"]["filename_or_obj"][0])

                    if opt.t1_aid:
                        # Pass inputs to model and optimise: Forward loop
                        if opt.weighted_sampling == "cropped":
                            real_T1 = train_sample['T1'].cuda()
                        else:
                            real_T1 = train_sample[0]['T1'].cuda()
                        fake_B = G_A(torch.cat((real_A, train_coords), dim=1))
                        # Pair fake B with fake z to generate rec_A: Add Coords as well and T1
                        rec_A = G_B(torch.cat((fake_B, real_T1, train_coords), dim=1))

                        # Backward loop: Sample z from normal distribution
                        fake_A = G_B(torch.cat((real_B, real_T1, train_coords), dim=1))
                        # Reconstructed B
                        if opt.cycle_noise:
                            rec_B = G_A(torch.cat((post_gen_noise(fake_A), train_coords), dim=1))
                        else:
                            rec_B = G_A(torch.cat((fake_A, train_coords), dim=1))
                    else:
                        # Pass inputs to model and optimise: Forward loop
                        fake_B = G_A(torch.cat((real_A, train_coords), dim=1))
                        # Pair fake B with fake z to generate rec_A: Add Coords as well
                        rec_A = G_B(torch.cat((fake_B, train_coords), dim=1))

                        # Backward loop: Sample z from normal distribution
                        fake_A = G_B(torch.cat((real_B, train_coords), dim=1))
                        # Reconstructed B
                        if opt.cycle_noise:
                            rec_B = G_A(torch.cat((post_gen_noise(fake_A), train_coords), dim=1))
                        else:
                            rec_B = G_A(torch.cat((fake_A, train_coords), dim=1))
                    # Training
                    # Only begin to train discriminator after model has started to converge
                    adv_start = time.time()
                    # D_A_total_loss = torch.zeros(1, )
                    # D_B_total_loss = torch.zeros(1, )
                    agg_real_D_A_acc = 0
                    agg_fake_D_A_acc = 0
                    agg_real_D_B_acc = 0
                    agg_fake_D_B_acc = 0
                    # Always have to do at least one run otherwise how is accuracy calculated?
                    for _ in range(1):
                        # Update discriminator by looping N times
                        # with torch.cuda.amp.autocast(enabled=True):
                        D_B_loss, real_D_B_acc, fake_D_B_acc, D_B_acc, _, fake_D_B_out = discriminator_loss(
                            gen_images=fake_B,
                            real_images=real_B,
                            discriminator=D_B,
                            real_label_flip_chance=opt.label_flipping_chance)
                        D_A_loss, real_D_A_acc, fake_D_A_acc, D_A_acc, _, fake_D_A_out = discriminator_loss(
                            gen_images=fake_A,
                            real_images=real_A,
                            discriminator=D_A,
                            real_label_flip_chance=opt.label_flipping_chance)

                        # if overall_disc_acc < disc_acc_thr_upper:
                        if train_D_A or train_D_B:
                            D_optimizer.zero_grad()

                        # Propagate + Log
                        if train_D_B:
                            D_B_loss.backward()
                        if train_D_A:
                            D_A_loss.backward()
                        if train_D_A or train_D_B:
                            D_optimizer.step()

                        # Log
                        # D_B_total_loss += D_B_loss.item()
                        # D_A_total_loss += D_A_loss.item()

                        # Aggregate accuracy: D_B
                        # agg_real_D_B_acc += real_D_B_acc
                        # agg_fake_D_B_acc += fake_D_B_acc
                        #
                        # # Aggregate accuracy: D_A
                        # agg_real_D_A_acc += real_D_A_acc
                        # agg_fake_D_A_acc += fake_D_A_acc
                        #
                        # # D_B
                        # agg_real_D_B_acc = agg_real_D_B_acc / 1
                        # agg_fake_D_B_acc = agg_fake_D_B_acc / 1
                        # overall_D_B_acc = (agg_real_D_B_acc + agg_fake_D_B_acc) / 2
                        #
                        # # D_A
                        # agg_real_D_A_acc = agg_real_D_A_acc / 1
                        # agg_fake_D_A_acc = agg_fake_D_A_acc / 1
                        # overall_D_A_acc = (agg_real_D_A_acc + agg_fake_D_A_acc) / 2

                    # Generator training
                    if train_G_A or train_G_B:
                        G_optimizer.zero_grad()

                    # with torch.cuda.amp.autocast(enabled=True):
                    # Train Generator: Always do this or make it threshold based as well?
                    if train_G_A:
                        G_A_loss, G_A_acc = generator_loss(gen_images=fake_B, discriminator=D_B)
                        A_cycle = criterionCycleA(rec_A, real_A)
                    if train_G_B:
                        G_B_loss, G_B_acc = generator_loss(gen_images=fake_A, discriminator=D_A)
                        if opt.block_cycle_B:
                            B_cycle = criterionCycleB(rec_B, real_B)

                    # Idt losses
                    # idt_A_loss = criterionIdt(idt_A, real_B)
                    # idt_B_loss = criterionIdt(idt_B, real_A)
                    if opt.seg_loss and train_G_B and epoch >= opt.vseg_epoch:
                        preproc = preprocessing(f"{base_dir}/MRA-GAN/VSeg/MAT_files", fake_A)
                        slog_fake_A = preproc.process()

                        # Output segmentation
                        seg_fake_A = torch.softmax(vseg_model(torch.cat((fake_A, slog_fake_A[None, None, ...].cuda()),
                                                                        dim=1)), dim=1)

                        # Loss
                        loss_seg_fake_A_loss = opt.vseg_loss_scaling * criterionDice(seg_fake_A[:, 1, ...][:, None, ...], real_B)

                        # Save, sometimes
                        if running_iter % 200 == 0:
                            save_img(seg_fake_A[:, 1, ...].squeeze().cpu().detach().numpy(),
                                     train_sample[0]['image_meta_dict']['affine'][0, ...],
                                     os.path.join(FIG_DIR, f"seg_fake_A_iter_{running_iter}.nii.gz"),
                                     overwrite=True)
                            save_img(real_B.squeeze().cpu().detach().numpy(),
                                     train_sample[0]['image_meta_dict']['affine'][0, ...],
                                     os.path.join(FIG_DIR, f"seg_real_B_iter_{running_iter}.nii.gz"),
                                     overwrite=True)
                        del seg_fake_A, slog_fake_A

                    # Total loss
                    if opt.perceptual and not opt.msssim:
                        if train_G_A:
                            A_perceptual_loss = perceptual_loss(real_A, rec_A, perceptual_net,
                                                                opt.patch_size) * opt.perceptual_weighting
                            total_G_A_loss = G_A_loss + A_cycle + A_perceptual_loss
                        if train_G_B:
                            if opt.block_cycle_B:
                                total_G_B_loss = G_B_loss
                            total_G_B_loss = G_B_loss + B_cycle
                            # total_G_loss = G_A_loss + A_cycle + A_perceptual_loss + G_B_loss + B_cycle
                    elif not opt.perceptual and opt.msssim:
                        if train_G_A:
                            A_msssim_loss = criterionMSSSIM(real_A, rec_A) * opt.msssim_weighting
                            total_G_A_loss = G_A_loss + A_cycle + A_msssim_loss
                        if train_G_B:
                            if opt.block_cycle_B:
                                total_G_B_loss = G_B_loss
                            total_G_B_loss = G_B_loss + B_cycle
                            # total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle + A_msssim_loss
                    elif opt.perceptual and opt.msssim:
                        if train_G_A:
                            A_perceptual_loss = perceptual_loss(real_A, rec_A, perceptual_net,
                                                                opt.patch_size) * opt.perceptual_weighting
                            A_msssim_loss = criterionMSSSIM(real_A, rec_A) * opt.msssim_weighting
                            total_G_A_loss = G_A_loss + A_cycle + A_msssim_loss + A_perceptual_loss
                        if train_G_B:
                            if opt.block_cycle_B:
                                total_G_B_loss = G_B_loss
                        total_G_B_loss = G_B_loss + B_cycle
                        # total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle + A_msssim_loss + A_perceptual_loss
                    else:
                        if train_G_A:
                            total_G_A_loss = G_A_loss + A_cycle
                        if train_G_B:
                            if opt.block_cycle_B:
                                total_G_B_loss = G_B_loss
                            total_G_B_loss = G_B_loss + B_cycle
                            # total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle
                    if opt.seg_loss and train_G_B and epoch >= opt.vseg_epoch:
                        total_G_B_loss = total_G_B_loss + loss_seg_fake_A_loss

                    # Backward
                    if train_G_A:
                        total_G_A_loss.backward()
                    if train_G_B:
                        total_G_B_loss.backward()
                    # total_G_loss.backward()

                    # G optimization
                    if train_G_A or train_G_B:
                        G_optimizer.step()

                    if opt.perceptual and opt.msssim:
                        print(f"Percep: {A_perceptual_loss.cpu().detach().tolist():.3f}, "
                              f"MS-SSIM: {A_msssim_loss.cpu().detach().tolist()[0][0]:.3f}, "
                              f"G_A: {G_A_loss.cpu().detach().tolist():.3f}, "
                              f"G_B: {G_B_loss.cpu().detach().tolist():.3f}, "
                              f"cycle_A: {A_cycle.cpu().detach().tolist():.3f}")
                        if not opt.block_cycle_B:
                            print(f", cycle_B: {B_cycle.cpu().detach().tolist():.3f}")

                    if running_iter % 100 == 0:
                        print(f"\nIteration: {running_iter}")
                        print(f"D_A Real Acc: {real_D_A_acc:.3f}")
                        print(f"D_B Real Acc: {real_D_B_acc:.3f}")
                        print(f"D_A Fake Acc: {fake_D_A_acc:.3f}")
                        print(f"D_B Fake Acc: {fake_D_B_acc:.3f}")
                        if opt.seg_loss and train_G_B and epoch >= opt.vseg_epoch:
                            print(f"Seg Loss: {loss_seg_fake_A_loss:.3f}")

                    # if total_steps % opt.print_freq == 0:
                    if total_steps % opt.save_latest_freq == 0:
                        print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                        # Saving
                        # Define ONE file for saving ALL state dicts
                        G_A.cpu()
                        G_B.cpu()
                        D_A.cpu()
                        D_B.cpu()
                        save_filename = f'epoch_{epoch + 1}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
                        current_state_dict = {
                            'G_optimizer_state_dict': G_optimizer.state_dict(),
                            'D_optimizer_state_dict': D_optimizer.state_dict(),
                            'G_scheduler_state_dict': G_scheduler.state_dict(),
                            'D_scheduler_state_dict': D_scheduler.state_dict(),
                            'epoch': epoch + 1,
                            'running_iter': running_iter,
                            'total_steps': total_steps,
                            'batch_size': opt.batch_size,
                            'patch_size': opt.patch_size,
                            'G_A_state_dict': G_A.state_dict(),
                            'G_B_state_dict': G_B.state_dict(),
                            'D_A_state_dict': D_A.state_dict(),
                            'D_B_state_dict': D_B.state_dict(),
                        }

                        # Actually save
                        torch.save(current_state_dict, os.path.join(MODELS_DIR,
                                                                    save_filename))

                        G_A.cuda()
                        G_B.cuda()
                        D_A.cuda()
                        D_B.cuda()

                    if running_iter % logging_interval == 0:
                        # Graphs
                        if opt.adversarial_loss_plot:
                            loss_adv_dict = {  # "Total_G_loss": total_G_loss,
                                "Generator_A": G_A_loss,
                                "Generator_B": G_B_loss,
                                "Discriminator_A": D_A_loss,
                                "Discriminator_B": D_B_loss,
                            }
                            loss_granular_dict = {  # "total_G_loss": total_G_loss,
                                "Generator_A": G_A_loss,
                                "Generator_B": G_B_loss,
                                "cycle_A": A_cycle,
                            }
                            if not opt.block_cycle_B:
                                loss_granular_dict["cycle_B"] = B_cycle
                        else:
                            loss_adv_dict = {
                                "Generator_A": G_A_acc,
                                "Generator_B": G_B_acc,
                                "Discriminator_A": D_A_acc,
                                "Discriminator_B": D_B_acc,
                            }
                            loss_granular_dict = {
                                "Generator_A": G_A_acc,
                                "Generator_B": G_B_acc,
                                "Discriminator_A_real": real_D_A_acc,
                                "Discriminator_A_fake": fake_D_A_acc,
                                "Discriminator_B_real": real_D_B_acc,
                                "Discriminator_B_fake": fake_D_B_acc,
                                "cycle_A": A_cycle,
                            }
                            if not opt.block_cycle_B:
                                loss_granular_dict["cycle_B"] = B_cycle
                        if opt.perceptual:
                            loss_adv_dict["Perceptual_A"] = A_perceptual_loss
                            loss_granular_dict["Perceptual_A"] = A_perceptual_loss
                        if opt.msssim:
                            loss_adv_dict["MSSSIM_A"] = A_msssim_loss
                            loss_granular_dict["MSSSIM_A"] = A_msssim_loss
                        if opt.seg_loss and train_G_B and epoch > opt.vseg_epoch:
                            loss_adv_dict["Seg_loss"] = loss_seg_fake_A_loss

                        writer.add_scalars('Loss/Adversarial',
                                           loss_adv_dict, running_iter)
                        writer.add_scalars('Loss/Granular_G',
                                           loss_granular_dict, running_iter)

                        # Slim writer only contains training curves
                        slim_writer.add_scalars('Loss/Adversarial',
                                                loss_adv_dict, running_iter)
                        slim_writer.add_scalars('Loss/Granular_G',
                                                loss_granular_dict, running_iter)

                        # Images
                        # Reals
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             real_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Visuals/Real_B_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             real_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Visuals/Real_A_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)

                        if opt.t1_aid:
                            img2tensorboard.add_animated_gif(writer=writer,
                                                             image_tensor=normalise_images(
                                                                 real_T1[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                             tag=f'Visuals/T1_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)

                        # Generated
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             fake_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Visuals/Fake_B_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             fake_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Visuals/Fake_A_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             rec_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Visuals/Rec_B_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             rec_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Visuals/Rec_A_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                    iter_data_time = time.time()
                    running_iter += 1

                    # Clean-up
                    del real_A, real_B, train_sample, rec_A, rec_B, train_coords, fake_D_A_out, fake_D_B_out
                    # del D_A_loss, D_B_loss, G_A_loss, G_B_loss, A_cycle, B_cycle
                    if opt.perceptual and train_G_A:
                        del A_perceptual_loss
                    if opt.msssim and train_G_A:
                        del A_msssim_loss
                    if opt.t1_aid:
                        del real_T1
                    if opt.seg_loss and train_G_B and epoch >= opt.vseg_epoch:
                        del loss_seg_fake_A_loss
                    if train_G_B:
                        del total_G_B_loss
                    if train_G_A:
                        del total_G_A_loss

                    # import gc
                    #
                    # for obj in gc.get_objects():
                    #     try:
                    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    #             print(type(obj), obj.size())
                    #     except:
                    #         pass

                if epoch % val_gap == 0:
                    G_A.eval()
                    G_B.eval()
                    D_A.eval()
                    D_B.eval()

                    val_agg_real_D_A_acc = []
                    val_agg_fake_D_A_acc = []
                    val_agg_real_D_B_acc = []
                    val_agg_fake_D_B_acc = []

                    val_agg_D_A_acc = []
                    val_agg_D_B_acc = []

                    val_agg_G_A_acc = []
                    val_agg_G_B_acc = []
                    with torch.no_grad():
                        for val_iter, val_sample in enumerate(val_loader):
                            # Validation variables
                            if opt.weighted_sampling == "cropped":
                                val_real_A = val_sample['image'].cuda()
                                val_real_B = val_sample['label'].cuda()
                                val_coords = val_sample['coords'].cuda()

                                image_name = os.path.basename(val_sample["image_meta_dict"]["filename_or_obj"][0])
                                label_name = os.path.basename(val_sample["label_meta_dict"]["filename_or_obj"][0])
                                val_affine = val_sample['image_meta_dict']['affine'][0, ...]
                                label_affine = val_sample['label_meta_dict']['affine'][0, ...]
                            else:
                                val_real_A = val_sample[0]['image'].cuda()
                                val_real_B = val_sample[0]['label'].cuda()
                                val_coords = val_sample[0]['coords'].cuda()

                                image_name = os.path.basename(val_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                                label_name = os.path.basename(val_sample[0]["label_meta_dict"]["filename_or_obj"][0])
                                val_affine = val_sample[0]['image_meta_dict']['affine'][0, ...]
                                label_affine = val_sample[0]['label_meta_dict']['affine'][0, ...]

                            # Forward
                            if opt.t1_aid:
                                if opt.weighted_sampling == "cropped":
                                    val_real_T1 = val_sample['T1'].cuda()
                                else:
                                    val_real_T1 = val_sample[0]['T1'].cuda()
                                # Pass inputs to model and optimise: Forward loop
                                val_fake_B = G_A(torch.cat((val_real_A, val_coords), dim=1))
                                # Pair fake B with fake z to generate rec_A
                                val_rec_A = G_B(torch.cat((val_fake_B, val_real_T1, val_coords), dim=1))

                                # Backward loop
                                val_fake_A = G_B(torch.cat((val_real_B, val_real_T1, val_coords), dim=1))
                                # Reconstructed B
                                if opt.cycle_noise:
                                    val_rec_B = G_A(torch.cat((post_gen_noise(val_fake_A), val_coords), dim=1))
                                else:
                                    val_rec_B = G_A(torch.cat((val_fake_A, val_coords), dim=1))
                            else:
                                # Pass inputs to model and optimise: Forward loop
                                val_fake_B = G_A(torch.cat((val_real_A, val_coords), dim=1))
                                # Pair fake B with fake z to generate rec_A
                                val_rec_A = G_B(torch.cat((val_fake_B, val_coords), dim=1))

                                # Backward loop
                                val_fake_A = G_B(torch.cat((val_real_B, val_coords), dim=1))
                                # Reconstructed B
                                if opt.cycle_noise:
                                    val_rec_B = G_A(torch.cat((post_gen_noise(val_fake_A), val_coords), dim=1))
                                else:
                                    val_rec_B = G_A(torch.cat((val_fake_A, val_coords), dim=1))

                            # "Losses"
                            # Discriminator
                            val_D_B_loss, val_real_D_B_acc, val_fake_D_B_acc, val_D_B_acc, _, val_fake_D_B_out = discriminator_loss(
                                gen_images=val_fake_B,
                                real_images=val_real_B,
                                discriminator=D_B,
                                real_label_flip_chance=0.0)
                            val_D_A_loss, val_real_D_A_acc, val_fake_D_A_acc, val_D_A_acc, _, val_fake_D_A_out = discriminator_loss(
                                gen_images=val_fake_A,
                                real_images=val_real_A,
                                discriminator=D_A,
                                real_label_flip_chance=0.0)

                            # Generator
                            val_G_A_loss, val_G_A_acc = generator_loss(gen_images=val_fake_B, discriminator=D_B)
                            val_G_B_loss, val_G_B_acc = generator_loss(gen_images=val_fake_A, discriminator=D_A)

                            # Aggregate accuracies
                            # Aggregate accuracy: D_B
                            val_agg_real_D_B_acc.append(real_D_B_acc)
                            val_agg_fake_D_B_acc.append(fake_D_B_acc)
                            val_agg_D_B_acc.append(val_D_B_acc)

                            # Aggregate accuracy: D_A
                            val_agg_real_D_A_acc.append(real_D_A_acc)
                            val_agg_fake_D_A_acc.append(fake_D_A_acc)
                            val_agg_D_A_acc.append(val_D_A_acc)

                            # Aggregate accuracy: G_A and G_B
                            val_agg_G_A_acc.append(val_G_A_acc)
                            val_agg_G_B_acc.append(val_G_B_acc)

                            # Cycle losses: G_A and G_B
                            val_A_cycle = criterionCycleA(val_rec_A, val_real_A)
                            val_B_cycle = criterionCycleB(val_rec_B, val_real_B)

                            if opt.seg_loss and epoch >= opt.vseg_epoch:
                                preproc = preprocessing(f"{base_dir}/MRA-GAN/VSeg/MAT_files", val_fake_A)
                                val_slog_fake_A = preproc.process()

                                # Output segmentation
                                val_seg_fake_A = torch.softmax(
                                    vseg_model(torch.cat((val_fake_A, val_slog_fake_A[None, None, ...].cuda()),
                                                         dim=1)), dim=1)

                                # Loss
                                val_loss_seg_fake_A_loss = criterionDice(val_seg_fake_A[:, 1, ...][:, None, ...],
                                                                         val_real_B)

                                # Save, sometimes
                                if val_iter % 200 == 0:
                                    save_img(val_seg_fake_A[:, 1, ...].squeeze().cpu().detach().numpy(),
                                             val_affine,
                                             os.path.join(FIG_DIR, f"val_seg_fake_A_iter_{running_iter}.nii.gz"),
                                             overwrite=True)
                                    save_img(val_real_B.squeeze().cpu().detach().numpy(),
                                             val_affine,
                                             os.path.join(FIG_DIR, f"val_real_B_iter_{running_iter}.nii.gz"),
                                             overwrite=True)
                                del val_seg_fake_A, val_slog_fake_A

                            if opt.perceptual and not opt.msssim:
                                val_A_perceptual_loss = perceptual_loss(val_real_A, val_rec_A, perceptual_net,
                                                                        opt.patch_size) * opt.perceptual_weighting
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle + val_A_perceptual_loss
                            elif not opt.perceptual and opt.msssim:
                                val_A_msssim_loss = criterionMSSSIM(val_real_A, val_rec_A) * opt.msssim_weighting
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle + val_A_msssim_loss
                            elif opt.perceptual and opt.msssim:
                                val_A_perceptual_loss = perceptual_loss(val_real_A, val_rec_A, perceptual_net,
                                                                        opt.patch_size) * opt.perceptual_weighting
                                val_A_msssim_loss = criterionMSSSIM(val_real_A, val_rec_A) * opt.msssim_weighting
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle + val_A_msssim_loss + val_A_perceptual_loss
                            else:
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle

                        # Graphs
                        if opt.adversarial_loss_plot:
                            val_loss_adv_dict = {
                                "Generator_A": val_G_A_loss,
                                "Generator_B": val_G_B_loss,
                                "Discriminator_A": val_D_A_loss,
                                "Discriminator_B": val_D_B_loss,
                            }
                            val_loss_granular_dict = {
                                "Generator_A": val_G_A_loss,
                                "Generator_B": val_G_B_loss,
                                "Discriminator_A": val_D_A_loss,
                                "Discriminator_B": val_D_B_loss,
                            }
                        else:
                            val_loss_adv_dict = {
                                "Generator_A": np.mean(val_agg_G_A_acc),
                                "Generator_B": np.mean(val_agg_G_B_acc),
                                "Discriminator_A": np.mean(val_agg_D_A_acc),
                                "Discriminator_B": np.mean(val_agg_D_B_acc),
                            }
                            val_loss_granular_dict = {
                                "Generator_A": np.mean(val_agg_G_A_acc),
                                "Generator_B": np.mean(val_agg_G_B_acc),
                                "Discriminator_A_real": np.mean(val_agg_real_D_A_acc),
                                "Discriminator_A_fake": np.mean(val_agg_fake_D_A_acc),
                                "Discriminator_B_real": np.mean(val_agg_real_D_B_acc),
                                "Discriminator_B_fake": np.mean(val_agg_fake_D_B_acc),
                            }
                        if opt.perceptual:
                            val_loss_adv_dict["Perceptual_A"] = val_A_perceptual_loss
                            val_loss_granular_dict["Perceptual_A"] = val_A_perceptual_loss
                        if opt.msssim:
                            val_loss_adv_dict["MSSSIM_A"] = val_A_msssim_loss
                            val_loss_granular_dict["MSSSIM_A"] = val_A_msssim_loss
                        if opt.seg_loss and epoch >= opt.vseg_epoch:
                            val_loss_adv_dict["Seg_loss"] = val_loss_seg_fake_A_loss
                        writer.add_scalars('Loss/Val_Adversarial',
                                           val_loss_adv_dict, running_iter)

                        writer.add_scalars('Loss/Val_Granular_G',
                                           val_loss_granular_dict, running_iter)

                        # Slim writer only contains training curves
                        slim_writer.add_scalars('Loss/Val_Adversarial',
                                                val_loss_adv_dict, running_iter)

                        slim_writer.add_scalars('Loss/Val_Granular_G',
                                                val_loss_granular_dict, running_iter)

                        # Saving
                        print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                        # Saving
                        # Define ONE file for saving ALL state dicts
                        G_A.cpu()
                        G_B.cpu()
                        D_A.cpu()
                        D_B.cpu()
                        save_filename = f'epoch_{epoch + 1}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
                        current_state_dict = {
                            'G_optimizer_state_dict': G_optimizer.state_dict(),
                            'D_optimizer_state_dict': D_optimizer.state_dict(),
                            'G_scheduler_state_dict': G_scheduler.state_dict(),
                            'D_scheduler_state_dict': D_scheduler.state_dict(),
                            'epoch': epoch + 1,
                            'running_iter': running_iter,
                            'total_steps': total_steps,
                            'batch_size': opt.batch_size,
                            'patch_size': opt.patch_size,
                            'G_A_state_dict': G_A.state_dict(),
                            'G_B_state_dict': G_B.state_dict(),
                            'D_A_state_dict': D_A.state_dict(),
                            'D_B_state_dict': D_B.state_dict(),
                        }

                        # Actually save
                        torch.save(current_state_dict, os.path.join(MODELS_DIR,
                                                                    save_filename))

                        G_A.cuda()
                        G_B.cuda()
                        D_A.cuda()
                        D_B.cuda()

                        # Images
                        # Reals
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             val_real_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Validation/Real_B_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             val_real_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Validation/Real_A_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)

                        if opt.t1_aid:
                            img2tensorboard.add_animated_gif(writer=writer,
                                                             image_tensor=normalise_images(
                                                                 val_real_T1[0, 0, ...][
                                                                     None, ...].cpu().detach().numpy()),
                                                             tag=f'Validation/T1_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)

                        # Generated
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             val_fake_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Validation/Fake_B_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             val_fake_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Validation/Fake_A_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             val_rec_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Validation/Rec_B_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)
                        img2tensorboard.add_animated_gif(writer=writer,
                                                         image_tensor=normalise_images(
                                                             val_rec_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                         tag=f'Validation/Rec_A_fold_{fold}',
                                                         max_out=opt.patch_size // 4,
                                                         scale_factor=255, global_step=running_iter)

                        # Clean-up
                        del val_real_A, val_real_B, val_sample, val_rec_A, val_rec_B
                        if opt.t1_aid:
                            del val_real_T1
                        if opt.seg_loss and epoch >= opt.vseg_epoch:
                            del val_loss_seg_fake_A_loss

                print(f'End of epoch {epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
                G_scheduler.step(epoch)
                D_scheduler.step(epoch)
        elif opt.phase == "test":
            # Overlap
            overlap = 0.0

            # Carry out inference
            G_A.eval()
            G_B.eval()
            with torch.no_grad():
                for inf_sample in inf_loader:
                    inf_real_A = inf_sample['image'].cuda()
                    inf_real_B = inf_sample['label'].cuda()
                    inf_coords = inf_sample['coords'].cuda()

                    image_name = os.path.basename(inf_sample["image_meta_dict"]["filename_or_obj"][0])
                    label_name = os.path.basename(inf_sample["label_meta_dict"]["filename_or_obj"][0])
                    inf_affine = inf_sample['image_meta_dict']['affine'][0, ...]
                    label_affine = inf_sample['label_meta_dict']['affine'][0, ...]

                    if opt.t1_aid:
                        # Pass inputs to generators
                        inf_real_T1 = inf_sample['T1'].cuda()
                        t1_name = os.path.basename(inf_sample["T1_meta_dict"]["filename_or_obj"][0])
                        fake_B = sliding_window_inference(torch.cat((inf_real_A, inf_coords), dim=1), (192, 224, 192),
                                                          1,
                                                          G_A,
                                                          overlap=overlap,
                                                          mode='gaussian')
                        fake_A = sliding_window_inference(torch.cat((inf_real_B, inf_real_T1, inf_coords), dim=1),
                                                          (192, 224, 192),
                                                          1,
                                                          G_B,
                                                          overlap=overlap,
                                                          mode='gaussian')

                        rec_A = sliding_window_inference(torch.cat((fake_B, inf_real_T1, inf_coords), dim=1),
                                                         (192, 224, 192), 1,
                                                         G_B,
                                                         overlap=overlap,
                                                         mode='gaussian')
                        rec_B = sliding_window_inference(torch.cat((fake_A, inf_coords), dim=1), (192, 224, 192), 1,
                                                         G_A,
                                                         overlap=overlap,
                                                         mode='gaussian')

                        print(f"The MRA, vasculature, and T1 names are: {image_name}, {label_name}, {t1_name}")

                    else:
                        if opt.weighted_sampling == "cropped":
                            inf_spatial_size = (192, 224, 128)
                            # Pass inputs to generators
                            fake_B = simple_inferer(torch.cat((inf_real_A, inf_coords), dim=1),
                                                    G_A)
                            fake_A = simple_inferer(torch.cat((inf_real_B, inf_coords), dim=1),
                                                    G_B)

                            rec_A = simple_inferer(torch.cat((fake_B, inf_coords), dim=1),
                                                   G_B)
                            rec_B = simple_inferer(torch.cat((fake_A, inf_coords), dim=1),
                                                   G_A)
                        else:
                            inf_spatial_size = (192, 224, 192)
                            # Pass inputs to generators
                            fake_B = sliding_window_inference(torch.cat((inf_real_A, inf_coords), dim=1),
                                                              inf_spatial_size,
                                                              1, G_A,
                                                              overlap=overlap,
                                                              mode='gaussian')
                            fake_A = sliding_window_inference(torch.cat((inf_real_B, inf_coords), dim=1),
                                                              inf_spatial_size,
                                                              1, G_B,
                                                              overlap=overlap,
                                                              mode='gaussian')

                            rec_A = sliding_window_inference(torch.cat((fake_B, inf_coords), dim=1), inf_spatial_size,
                                                             1,
                                                             G_B,
                                                             overlap=overlap,
                                                             mode='gaussian')
                            rec_B = sliding_window_inference(torch.cat((fake_A, inf_coords), dim=1), inf_spatial_size,
                                                             1,
                                                             G_A,
                                                             overlap=overlap,
                                                             mode='gaussian')

                        print(f"The MRA and vasculature names are: {image_name}, {label_name}")

                    del inf_real_A, inf_real_B, inf_sample, inf_coords
                    if opt.t1_aid:
                        del inf_real_T1
                        # Need to know which T1 was used for generation!
                        fake_image_basename = f"Vasc_{basename_extractor(label_name, keep_extension=False)}_" \
                                              f"T1_{basename_extractor(t1_name, keep_extension=True)}"
                    else:
                        fake_image_basename = f"Vasc_{basename_extractor(label_name, keep_extension=True)}"
                    fake_vasc_basename = f"Image_{basename_extractor(image_name, keep_extension=True)}"

                    # Saving: Fakes
                    save_img(normalise_images(fake_A.cpu().detach().squeeze().numpy()),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_A_" + fake_image_basename),
                             overwrite=True)  # MRA
                    # # Add noise
                    # fake_B = torch.abs(post_gen_noise(fake_B))
                    save_img(normalise_images(fake_B.cpu().detach().squeeze().numpy()),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_B_" + fake_vasc_basename),
                             overwrite=True)

                    # Saving: Reconstructions
                    save_img(normalise_images(rec_A.cpu().detach().squeeze().numpy()),
                             inf_affine,
                             os.path.join(FIG_DIR, "Rec_A_" + fake_image_basename),
                             overwrite=True)
                    save_img(normalise_images(rec_B.cpu().detach().squeeze().numpy()),
                             inf_affine,
                             os.path.join(FIG_DIR, "Rec_B_" + fake_vasc_basename),
                             overwrite=True)
