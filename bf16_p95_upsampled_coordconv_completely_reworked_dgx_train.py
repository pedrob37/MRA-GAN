import os
import torch
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
import time
from models import create_model
from utils.visualizer import Visualizer
from utils.utils import create_path, save_img, CoordConvd, kernel_size_calculator
import monai
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from models.pix2pix_disc_networks import NoisyMultiscaleDiscriminator3D, GANLoss
import monai.visualize.img2tensorboard as img2tensorboard
from monai.transforms import (Compose,
                              LoadImaged,
                              AddChanneld,
                              RandAffined,
                              RandCropByPosNegLabeld,
                              RandBiasFieldd,
                              ToTensord,
                              RandSpatialCropSamplesd,
                              RandGaussianSmoothd,
                              RandGaussianNoiseD,
                              SpatialPadd,
                              )
from monai.utils import set_determinism
from utils.MSSSIM.pytorch_msssim.ssim import SSIM


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    # DDP functionality
    import deepspeed

    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=True,
        verbose=False,
        init_method=None,
        distributed_port=opt.master_port_id,
    )

    # DDP variables: global rank, local rank, number of processes
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f'Current local rank is {local_rank}')
    print(f"Current rank is {rank}")
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "COLL"

    # Set determinism: Randomly pick a number, but log it!
    random_seed_number = np.random.randint(0, 100, 1)[0]
    set_determinism(random_seed_number + local_rank)
    print(f"The random seed is {random_seed_number}")

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_number
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # chin added 2022.01.28
    # Determine device
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Define list to iterate through for logging: This process as to be deterministic otherwise risk multi-logging!
    from itertools import cycle
    cycled_ranks = cycle(list(range(world_size)))

    # Model choice: Trilinear, nearest neighbour, or nearest neighbour + subpixel convolution
    if rank == 0:
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

    # Read csv + add directory to filenames
    df = pd.read_csv(opt.csv_file)
    df['Label_Filename'] = df['Filename']
    df['Filename'] = images_dir + '/' + df['Filename'].astype(str)
    df['Label_Filename'] = labels_dir + '/' + df['Label_Filename'].astype(str)
    num_folds = df.fold.nunique()

    # Inference fold assignment
    inf_fold = 9
    inf_df = df[df.fold == inf_fold]
    inf_df.reset_index(drop=True, inplace=True)

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
    criterionCycle = torch.nn.L1Loss().to(dtype=torch.bfloat16)
    criterionIdt = torch.nn.L1Loss().to(dtype=torch.bfloat16)

    # MS-SSIM
    if opt.msssim:
        gaussian_kernel_size = kernel_size_calculator(opt.patch_size)
        criterionMSSSIM = SSIM(data_range=1.0,
                               gaussian_kernel_size=gaussian_kernel_size,
                               gradient_based=True,
                               star_based=False)

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
        real_disc_sum_ds1 = real_disc_prediction[0][0].float().sum(axis=(1, 2, 3, 4)) / real_disc_prediction[0][0][
            0, ...].nelement()
        fake_disc_sum_ds1 = fake_disc_prediction[0][0].float().sum(axis=(1, 2, 3, 4)) / fake_disc_prediction[0][0][
            0, ...].nelement()
        # real_disc_sum_ds2 = real_disc_prediction[1][0].float().sum(axis=(1, 2, 3, 4)) / real_disc_prediction[1][0][
        #     0, ...].nelement()
        # fake_disc_sum_ds2 = fake_disc_prediction[1][0].float().sum(axis=(1, 2, 3, 4)) / fake_disc_prediction[1][0][
        #     0, ...].nelement()

        real_disc_accuracy_ds1 = ((real_disc_sum_ds1 > 0.5) == real_label).float().sum() / real_images.shape[0]
        fake_disc_accuracy_ds1 = ((fake_disc_sum_ds1 > 0.5) == fake_label).float().sum() / real_images.shape[0]
        # real_disc_accuracy_ds2 = ((real_disc_sum_ds2 > 0.5) == real_label).float().sum() / real_images.shape[0]
        # fake_disc_accuracy_ds2 = ((fake_disc_sum_ds2 > 0.5) == fake_label).float().sum() / real_images.shape[0]

        # (real_disc_accuracy_ds1 + real_disc_accuracy_ds2) / 2, \
        # (fake_disc_accuracy_ds1 + fake_disc_accuracy_ds2) / 2, \

        return (genloss + realloss) / 2, \
               real_disc_accuracy_ds1, \
               fake_disc_accuracy_ds1, \
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
        return gen_fake_loss

    # Augmentations/ Transforms
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
                                        RandGaussianNoiseD(keys=["image"], std=0.2, prob=0.5),
                                        RandSpatialCropSamplesd(keys=["image", "label", "coords"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label', 'coords'])])
        elif opt.augmentation_level == "light":
            # TODO list transforms for easy composition with flags
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
                                        RandSpatialCropSamplesd(keys=["image", "label", "coords"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label', 'coords'])])
        elif opt.augmentation_level == "none":
            train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                        RandSpatialCropSamplesd(keys=["image", "label", "coords"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label', 'coords'])])

        val_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                  AddChanneld(keys=['image', 'label']),
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                  RandSpatialCropSamplesd(keys=["image", "label", "coords"],
                                                          roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                          random_center=True,
                                                          random_size=False,
                                                          num_samples=1),
                                  # SpatialPadd(keys=["image", "label"],
                                  #             spatial_size=opt.patch_size),
                                  ToTensord(keys=['image', 'label', 'coords'])])
    elif opt.phase == "test":
        from monai.inferers import sliding_window_inference
        inf_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                  AddChanneld(keys=['image', 'label']),
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                  # RandSpatialCropSamplesd(keys=["image", "label"],
                                  #                         roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                  #                         random_center=True,
                                  #                         random_size=False,
                                  #                         num_samples=1),
                                  # SpatialPadd(keys=["image", "label"],
                                  #             spatial_size=opt.patch_size),
                                  ToTensord(keys=['image', 'label', 'coords'])])

    ## Relevant job directories
    base_dir = opt.base_dir
    if opt.phase == "train":
        # Main directories
        CACHE_DIR = f"{base_dir}/Outputs-MRA-GAN/Cache/{opt.job_name}"
        FIG_DIR = f"{base_dir}/Outputs-MRA-GAN/Figures/{opt.job_name}"
        LOG_DIR = f'{base_dir}/Outputs-MRA-GAN/Logs/{opt.job_name}'
        MODELS_DIR = f"{base_dir}/Outputs-MRA-GAN/Models/{opt.job_name}"

        # Create directories
        create_path(CACHE_DIR)
        create_path(FIG_DIR)
        create_path(LOG_DIR)
        create_path(MODELS_DIR)
    elif opt.phase == "test" and opt.job_name.startswith("inf"):
        opt.job_name = opt.job_name.split('-', 1)[1]

        # Directories should already exist
        CACHE_DIR = f"{base_dir}/Outputs-MRA-GAN/Cache/{opt.job_name}"
        FIG_DIR = f"{base_dir}/Outputs-MRA-GAN/Figures/{opt.job_name}"
        LOG_DIR = f'{base_dir}/Outputs-MRA-GAN/Logs/{opt.job_name}'
        MODELS_DIR = f"{base_dir}/Outputs-MRA-GAN/Models/{opt.job_name}"
    else:
        raise NameError("Job phase is test but job name does not start with inf!")

    # Re-assign checkpoints directory
    opt.checkpoints_dir = MODELS_DIR
    opt.log_dir = LOG_DIR

    # Other variables
    val_gap = 5
    LOAD = True

    # Folds
    for fold in range(10):
        # Define models and associated variables
        if rank == 0:
            print('\nFOLD', fold)

        # Generators
        if opt.final_act == "leaky":
            G_A_final_act = torch.nn.LeakyReLU()
        elif opt.final_act == "sigmoid":
            G_A_final_act = torch.nn.Sigmoid()
        G_A = nnUNet(4, 1, dropout_level=0, final_act=G_A_final_act)
        G_B = nnUNet(4, 1, dropout_level=0)

        if opt.perceptual:
            import lpips
            perceptual_net = lpips.LPIPS(pretrained=True, net='squeeze')
            # if torch.cuda.device_count() > 1:
            perceptual_net = torch.nn.DataParallel(perceptual_net)
        # Discriminators
        D_A = NoisyMultiscaleDiscriminator3D(1, opt.ndf,
                                             opt.n_layers_D,
                                             nn.InstanceNorm3d, False, 1, False)

        D_B = NoisyMultiscaleDiscriminator3D(1, opt.ndf,
                                             opt.n_layers_D,
                                             nn.InstanceNorm3d, False, 1, False)

        # Passing to device
        G_A.to(device)
        G_B.to(device)
        D_A.to(device)
        D_B.to(device)

        # DDP
        G_A = torch.nn.parallel.DistributedDataParallel(
            G_A,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=12.5,
        ).to(dtype=torch.bfloat16)

        G_B = torch.nn.parallel.DistributedDataParallel(
            G_B,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=12.5,
        ).to(dtype=torch.bfloat16)

        D_A = torch.nn.parallel.DistributedDataParallel(
            D_A,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=12.5,
        ).to(dtype=torch.bfloat16)

        D_B = torch.nn.parallel.DistributedDataParallel(
            D_B,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=12.5,
        ).to(dtype=torch.bfloat16)

        # Optimizers + schedulers
        import itertools
        G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()),
                                       lr=opt.gen_lr, betas=(0.5, 0.999))
        D_optimizer = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                                       lr=opt.disc_lr, betas=(0.5, 0.999))

        # Domains
        G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, 0.995)
        D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, 0.995)

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
            checkpoint = torch.load(latest_model_file, map_location=torch.device("cuda", local_rank))
            print(f'Loading {latest_model_file}!')
            loaded_epoch = checkpoint['epoch']
            running_iter = checkpoint['running_iter']
            total_steps = checkpoint['total_steps']

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
            checkpoint = torch.load(latest_model_file, map_location=torch.device("cuda", local_rank))
            print(f'Loading {latest_model_file}!')
            loaded_epoch = checkpoint['epoch']
            running_iter = checkpoint['running_iter']
            total_steps = checkpoint['total_steps']

            # best_model_file = f'epoch_{loaded_epoch}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
            print(f'Loading checkpoint for model: {os.path.basename(latest_model_file)}')
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

        if opt.perceptual:
            perceptual_net = perceptual_net.cuda()

        # Train / Val split
        val_fold = fold
        excluded_folds = [val_fold, inf_fold]
        train_df = df[~df.fold.isin(excluded_folds)]
        val_df = df[df.fold == val_fold]
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        # Writer
        writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f'fold_{fold}'))

        # Print sizes
        print(f'The length of the training is {len(train_df)}')
        print(f'The length of the validation is {len(val_df)}')
        print(f'The length of the inference is {len(inf_df)}')

        # Data dicts
        train_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                           in zip(train_df.Filename, train_df.Label_Filename)]
        val_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                         in zip(val_df.Filename, val_df.Label_Filename)]
        inf_data_dict = [{'image': image_name, 'label': label_name} for image_name, label_name
                         in zip(inf_df.Filename, inf_df.Label_Filename)]

        # Basic length checks
        assert len(train_df.Filename) == len(train_df.Label_Filename)
        assert len(val_df.Filename) == len(val_df.Label_Filename)

        if opt.phase == "train":
            # Training + validation loaders
            train_ds = monai.data.PersistentDataset(data=train_data_dict,
                                                    transform=train_transforms,
                                                    cache_dir=CACHE_DIR
                                                    )

            train_loader = DataLoader(dataset=train_ds,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      num_workers=opt.workers,
                                      )

            val_ds = monai.data.PersistentDataset(data=val_data_dict,
                                                  transform=val_transforms,
                                                  cache_dir=CACHE_DIR
                                                  )

            val_loader = DataLoader(dataset=val_ds,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.workers,
                                    )
        elif opt.phase == "test":
            inf_ds = monai.data.Dataset(data=inf_data_dict,
                                        transform=inf_transforms,
                                        )

            inf_loader = DataLoader(dataset=inf_ds,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=opt.workers,
                                    )

        # Model creation
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
                for _, train_sample in enumerate(train_loader):
                    # Select a GPU for logging
                    chosen_rank = cycle(cycled_ranks)

                    iter_start_time = time.time()
                    if total_steps % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    visualizer.reset()

                    # Training variables: Global
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size

                    # Iteration-specific data
                    real_A = train_sample[0]['image'].cuda()
                    real_B = train_sample[0]['label'].cuda()
                    train_coords = train_sample[0]['coords'].cuda()

                    # Names (Not needed for now)
                    image_name = os.path.basename(train_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                    label_name = os.path.basename(train_sample[0]["label_meta_dict"]["filename_or_obj"][0])

                    # Pass inputs to model and optimise: Forward loop
                    fake_B = G_A(torch.cat((real_A, train_coords), dim=1))
                    # Pair fake B with fake z to generate rec_A: Add Coords as well
                    rec_A = G_B(torch.cat((fake_B, train_coords), dim=1))

                    # Backward loop: Sample z from normal distribution
                    fake_A = G_B(torch.cat((real_B, train_coords), dim=1))
                    # Reconstructed B
                    rec_B = G_A(torch.cat((fake_A, train_coords), dim=1))

                    # Identity
                    # idt_A = G_A(real_B)
                    # idt_B = G_B(real_A)

                    # Training
                    if epoch >= 0:
                        adv_start = time.time()
                        D_A_total_loss = torch.zeros(1,)
                        D_B_total_loss = torch.zeros(1,)
                        agg_real_D_A_acc = 0
                        agg_fake_D_A_acc = 0
                        agg_real_D_B_acc = 0
                        agg_fake_D_B_acc = 0
                        # Always have to do at least one run otherwise how is accuracy calculated?
                        for _ in range(1):
                            # Update discriminator by looping N times
                            # with torch.cuda.amp.autocast(enabled=True):
                            D_B_loss, real_D_B_acc, fake_D_B_acc, _, fake_D_B_out = discriminator_loss(gen_images=fake_B,
                                                                                                       real_images=real_B,
                                                                                                       discriminator=D_B,
                                                                                                       real_label_flip_chance=opt.label_flipping_chance)
                            D_A_loss, real_D_A_acc, fake_D_A_acc, _, fake_D_A_out = discriminator_loss(gen_images=fake_A,
                                                                                                       real_images=real_A,
                                                                                                       discriminator=D_A,
                                                                                                       real_label_flip_chance=opt.label_flipping_chance)

                            # if overall_disc_acc < disc_acc_thr_upper:
                            D_optimizer.zero_grad()

                            # Propagate + Log
                            D_B_loss.backward()
                            D_A_loss.backward()
                            D_optimizer.step()

                            # D_B_scaler.scale(D_B_loss).backward()
                            # D_B_scaler.step(D_B_optimizer)
                            # D_B_scaler.update()
                            D_B_total_loss += D_B_loss.item()
                            D_A_total_loss += D_A_loss.item()

                            # Aggregate accuracy: D_B
                            agg_real_D_B_acc += real_D_B_acc
                            agg_fake_D_B_acc += fake_D_B_acc

                            # Aggregate accuracy: D_A
                            agg_real_D_A_acc += real_D_A_acc
                            agg_fake_D_A_acc += fake_D_A_acc

                        # D_B
                        agg_real_D_B_acc = agg_real_D_B_acc / 1
                        agg_fake_D_B_acc = agg_fake_D_B_acc / 1
                        overall_D_B_acc = (agg_real_D_B_acc + agg_fake_D_B_acc) / 2

                        # D_A
                        agg_real_D_A_acc = agg_real_D_A_acc / 1
                        agg_fake_D_A_acc = agg_fake_D_A_acc / 1
                        overall_D_A_acc = (agg_real_D_A_acc + agg_fake_D_A_acc) / 2

                        # Generator training
                        G_optimizer.zero_grad()

                        # with torch.cuda.amp.autocast(enabled=True):
                        # Train Generator: Always do this or make it threshold based as well?
                        # Only update generator if discriminator is doing too well
                        G_A_loss = generator_loss(gen_images=fake_B, discriminator=D_B)
                        G_B_loss = generator_loss(gen_images=fake_A, discriminator=D_A)

                        # Cycle losses: G_A and G_B
                        A_cycle = criterionCycle(rec_A, real_A)
                        B_cycle = criterionCycle(rec_B, real_B)

                        # Idt losses
                        # idt_A_loss = criterionIdt(idt_A, real_B)
                        # idt_B_loss = criterionIdt(idt_B, real_A)

                        # Total loss
                        if opt.perceptual and not opt.msssim:
                            A_perceptual_loss = perceptual_loss(real_A, rec_A, perceptual_net, opt.patch_size) * opt.perceptual_weighting
                            total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle + A_perceptual_loss
                        elif not opt.perceptual and opt.msssim:
                            A_msssim_loss = criterionMSSSIM(real_A, rec_A)
                            total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle + A_msssim_loss
                        elif opt.perceptual and opt.msssim:
                            A_perceptual_loss = perceptual_loss(real_A, rec_A, perceptual_net, opt.patch_size) * opt.perceptual_weighting
                            A_msssim_loss = criterionMSSSIM(real_A, rec_A)
                            total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle + A_msssim_loss + A_perceptual_loss
                        else:
                            total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle

                        # Backward
                        total_G_loss.backward()

                        # G optimization
                        G_optimizer.step()

                    if rank == chosen_rank:
                        if total_steps % opt.save_latest_freq == 0:
                            print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                            # Saving
                            # Define ONE file for saving ALL state dicts
                            G_A.cpu()
                            G_B.cpu()
                            D_A.cpu()
                            D_B.cpu()
                            save_filename = f'epoch_{epoch+1}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
                            current_state_dict = {
                                'G_optimizer_state_dict': G_optimizer.state_dict(),
                                'D_optimizer_state_dict': D_optimizer.state_dict(),
                                'G_scheduler_state_dict': G_scheduler.state_dict(),
                                'D_scheduler_state_dict': D_scheduler.state_dict(),
                                'epoch': epoch+1,
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

                        # Logging: Only on a single GPU
                        if total_steps % 250 == 0:
                            # Graphs
                            loss_adv_dict = {"Total_G_loss": total_G_loss,
                                             "Generator_A": G_A_loss,
                                             "Generator_B": G_B_loss,
                                             "Discriminator_A": D_A_loss,
                                             "Discriminator_B": D_B_loss,
                                            }
                            loss_granular_dict = {"total_G_loss": total_G_loss,
                                                  "Generator_A": G_A_loss,
                                                  "Generator_B": G_B_loss,
                                                  "cycle_A": A_cycle,
                                                  "cycle_B": B_cycle,
                                                 }
                            if opt.perceptual:
                                loss_adv_dict["Perceptual_A"] = A_perceptual_loss
                                loss_granular_dict["Perceptual_A"] = A_perceptual_loss
                            if opt.msssim:
                                loss_adv_dict["MSSSIM_A"] = A_msssim_loss
                                loss_granular_dict["MSSSIM_A"] = A_msssim_loss

                                writer.add_scalars('Loss/Adversarial',
                                                   loss_adv_dict, running_iter)
                                writer.add_scalars('Loss/Granular_G',
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
                    del real_A, real_B, train_sample, rec_A, rec_B

                if epoch % val_gap == 0:
                    G_A.eval()
                    G_B.eval()
                    D_A.eval()
                    D_B.eval()
                    with torch.no_grad():
                        for val_sample in val_loader:
                            # Validation variables
                            val_real_A = val_sample[0]['image'].cuda()
                            val_real_B = val_sample[0]['label'].cuda()
                            val_coords = val_sample[0]['coords'].cuda()

                            image_name = os.path.basename(val_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                            label_name = os.path.basename(val_sample[0]["label_meta_dict"]["filename_or_obj"][0])
                            val_affine = val_sample[0]['image_meta_dict']['affine'][0, ...]
                            label_affine = val_sample[0]['label_meta_dict']['affine'][0, ...]

                            # Forward
                            # Pass inputs to model and optimise: Forward loop
                            val_fake_B = G_A(torch.cat((val_real_A, val_coords), dim=1))
                            # Pair fake B with fake z to generate rec_A
                            val_rec_A = G_B(torch.cat((val_fake_B, val_coords), dim=1))

                            # Backward loop
                            val_fake_A = G_B(torch.cat((val_real_B, val_coords), dim=1))
                            # Reconstructed B
                            val_rec_B = G_A(torch.cat((val_fake_A, val_coords), dim=1))

                            # Identity
                            # val_idt_A = G_A(val_real_B)
                            # val_idt_B = G_B(val_real_A)

                            # "Losses"
                            # Discriminator
                            val_D_B_loss, _, _, _, val_fake_D_B_out = discriminator_loss(gen_images=val_fake_B,
                                                                                         real_images=val_real_B,
                                                                                         discriminator=D_B,
                                                                                         real_label_flip_chance=0.0)
                            val_D_A_loss, _, _, _, val_fake_D_A_out = discriminator_loss(gen_images=val_fake_A,
                                                                                         real_images=val_real_A,
                                                                                         discriminator=D_A,
                                                                                         real_label_flip_chance=0.0)

                            # Generator
                            val_G_A_loss = generator_loss(gen_images=val_fake_B, discriminator=D_B)
                            val_G_B_loss = generator_loss(gen_images=val_fake_A, discriminator=D_A)

                            # Cycle losses: G_A and G_B
                            val_A_cycle = criterionCycle(val_rec_A, val_real_A)
                            val_B_cycle = criterionCycle(val_rec_B, val_real_B)

                            if opt.perceptual and not opt.msssim:
                                val_A_perceptual_loss = perceptual_loss(val_real_A, val_rec_A, perceptual_net, opt.patch_size) * opt.perceptual_weighting
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle + val_A_perceptual_loss
                            elif not opt.perceptual and opt.msssim:
                                val_A_msssim_loss = criterionMSSSIM(val_real_A, val_rec_A)
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle + val_A_msssim_loss
                            elif opt.perceptual and opt.msssim:
                                val_A_perceptual_loss = perceptual_loss(val_real_A, val_rec_A, perceptual_net, opt.patch_size) * opt.perceptual_weighting
                                val_A_msssim_loss = criterionMSSSIM(val_real_A, val_rec_A)
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle + val_A_msssim_loss + val_A_perceptual_loss
                            else:
                                val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle

                        # Graphs
                        if rank == chosen_rank:
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
                            if opt.perceptual:
                                val_loss_adv_dict["Perceptual_A"] = val_A_perceptual_loss
                                val_loss_granular_dict["Perceptual_A"] = val_A_perceptual_loss
                            if opt.msssim:
                                val_loss_adv_dict["MSSSIM_A"] = val_A_msssim_loss
                                val_loss_granular_dict["MSSSIM_A"] = val_A_msssim_loss
                            writer.add_scalars('Loss/Val_Adversarial',
                                               val_loss_adv_dict, running_iter)

                            writer.add_scalars('Loss/Val_Granular_G',
                                               val_loss_granular_dict, running_iter)

                            # Saving
                            print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                            # Saving
                            # Define ONE file for saving ALL state dicts
                            G_A.cpu()
                            G_B.cpu()
                            D_A.cpu()
                            D_B.cpu()
                            save_filename = f'epoch_{epoch+1}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
                            current_state_dict = {
                                'G_optimizer_state_dict': G_optimizer.state_dict(),
                                'D_optimizer_state_dict': D_optimizer.state_dict(),
                                'G_scheduler_state_dict': G_scheduler.state_dict(),
                                'D_scheduler_state_dict': D_scheduler.state_dict(),
                                'epoch': epoch+1,
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
                                                             image_tensor=normalise_images(val_real_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                             tag=f'Validation/Real_B_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)
                            img2tensorboard.add_animated_gif(writer=writer,
                                                             image_tensor=normalise_images(val_real_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                             tag=f'Validation/Real_A_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)

                            # Generated
                            img2tensorboard.add_animated_gif(writer=writer,
                                                             image_tensor=normalise_images(val_fake_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                             tag=f'Validation/Fake_B_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)
                            img2tensorboard.add_animated_gif(writer=writer,
                                                             image_tensor=normalise_images(val_fake_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                             tag=f'Validation/Fake_A_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)
                            img2tensorboard.add_animated_gif(writer=writer,
                                                             image_tensor=normalise_images(val_rec_B[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                             tag=f'Validation/Rec_B_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)
                            img2tensorboard.add_animated_gif(writer=writer,
                                                             image_tensor=normalise_images(val_rec_A[0, 0, ...][None, ...].cpu().detach().numpy()),
                                                             tag=f'Validation/Rec_A_fold_{fold}',
                                                             max_out=opt.patch_size // 4,
                                                             scale_factor=255, global_step=running_iter)

                        # Clean-up
                        del val_real_A, val_real_B, val_sample, val_rec_A, val_rec_B

                if rank == chosen_rank:
                    print(f'End of epoch {epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
                G_scheduler.step(epoch)
                D_scheduler.step(epoch)
        elif opt.phase == "test":
            # Overlap and sliding window inference
            from monai.inferers import sliding_window_inference
            overlap = 0.3

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

                    # Pass inputs to generators
                    fake_B = sliding_window_inference(torch.cat((inf_real_A, inf_coords), dim=1), 160, 1, G_A,
                                                      overlap=overlap,
                                                      mode='gaussian')
                    fake_A = sliding_window_inference(torch.cat((inf_real_B, inf_coords), dim=1), 160, 1, G_B,
                                                      overlap=overlap,
                                                      mode='gaussian')

                    rec_A = sliding_window_inference(torch.cat((fake_B, inf_coords), dim=1), 160, 1,
                                                     G_B,
                                                     overlap=overlap,
                                                     mode='gaussian')
                    rec_B = sliding_window_inference(torch.cat((fake_A, inf_coords), dim=1), 160, 1,
                                                     G_A,
                                                     overlap=overlap,
                                                     mode='gaussian')

                    del inf_real_A, inf_real_B, inf_sample, inf_coords

                    # Saving: Fakes
                    save_img(fake_A.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_A_" + os.path.basename(label_name)))
                    save_img(fake_B.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_B_" + os.path.basename(image_name)))

                    # Saving: Reconstructions
                    save_img(rec_A.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Rec_A_" + os.path.basename(image_name)))
                    save_img(rec_B.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Rec_B_" + os.path.basename(label_name)))
