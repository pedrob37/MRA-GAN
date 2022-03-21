import os
import torch
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
import time
from models import create_model
from utils.visualizer import Visualizer
from utils.utils import create_path, save_img
import monai
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from models.model_transconvs_norm import nnUNet
from models.pix2pix_disc_networks import NoisyMultiscaleDiscriminator3D, GANLoss
import monai.visualize.img2tensorboard as img2tensorboard
from models.resnets import resnet10
from monai.transforms import (Compose,
                              LoadImaged,
                              AddChanneld,
                              RandAffined,
                              RandCropByPosNegLabeld,
                              RandBiasFieldd,
                              ToTensord,
                              RandSpatialCropSamplesd,
                              NormalizeIntensityd,
                              RandGaussianSmoothd,
                              RandGaussianNoiseD,
                              SpatialPadd,
                              )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    ### MONAI
    # Data directory
    images_dir = os.path.join(opt.data_path, 'Images')  # Binary
    labels_dir = os.path.join(opt.data_path, 'Labels')  # MRA

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
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

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
                                        RandAffined(keys=["image", "label"],
                                                    scale_range=(0.1, 0.1, 0.1),
                                                    rotate_range=(0.25, 0.25, 0.25),
                                                    translate_range=(20, 20, 20),
                                                    mode=("bilinear", "nearest"),
                                                    as_tensor_output=False, prob=1.0,
                                                    padding_mode=('zeros', 'zeros')),
                                        RandGaussianSmoothd(keys=["image"], prob=0.25,  # 0.2
                                                            sigma_x=(0.25, 0.3),
                                                            sigma_y=(0.25, 0.3),
                                                            sigma_z=(0.25, 0.3)),
                                        RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.1, 0.25),
                                                       prob=0.25),  # Odd behaviour...
                                        NormalizeIntensityd(keys=['image'], channel_wise=True),
                                        RandGaussianNoiseD(keys=["image"], std=0.2, prob=0.5),
                                        RandSpatialCropSamplesd(keys=["image", "label"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label'])])
        elif opt.augmentation_level == "light":
            train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        RandAffined(keys=["image", "label"],
                                                    scale_range=(0.1, 0.1, 0.1),
                                                    rotate_range=(0.25, 0.25, 0.25),
                                                    translate_range=(20, 20, 20),
                                                    mode=("bilinear", "nearest"),
                                                    as_tensor_output=False, prob=1.0,
                                                    padding_mode=('zeros', 'zeros')),
                                        NormalizeIntensityd(keys=['image'], channel_wise=True),
                                        RandSpatialCropSamplesd(keys=["image", "label"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label'])])
        elif opt.augmentation_level == "none":
            train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        NormalizeIntensityd(keys=['image'], channel_wise=True),
                                        RandSpatialCropSamplesd(keys=["image", "label"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label'])])

        val_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                  AddChanneld(keys=['image', 'label']),
                                  NormalizeIntensityd(keys=['image'], channel_wise=True),
                                  RandSpatialCropSamplesd(keys=["image", "label"],
                                                          roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                          random_center=True,
                                                          random_size=False,
                                                          num_samples=1),
                                  # SpatialPadd(keys=["image", "label"],
                                  #             spatial_size=opt.patch_size),
                                  ToTensord(keys=['image', 'label'])])
    elif opt.phase == "test":
        from monai.inferers import sliding_window_inference
        inf_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                  AddChanneld(keys=['image', 'label']),
                                  # RandSpatialCropSamplesd(keys=["image", "label"],
                                  #                         roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                  #                         random_center=True,
                                  #                         random_size=False,
                                  #                         num_samples=1),
                                  # SpatialPadd(keys=["image", "label"],
                                  #             spatial_size=opt.patch_size),
                                  NormalizeIntensityd(keys=['image'], channel_wise=True),
                                  ToTensord(keys=['image', 'label'])])

    ## Relevant job directories
    if opt.phase == "train":
        # Main directories
        CACHE_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Cache/{opt.job_name}"
        FIG_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Figures/{opt.job_name}"
        LOG_DIR = f'/nfs/home/pedro/Outputs-MRA-GAN/Logs/{opt.job_name}'
        MODELS_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Models/{opt.job_name}"

        # Create directories
        create_path(CACHE_DIR)
        create_path(FIG_DIR)
        create_path(LOG_DIR)
        create_path(MODELS_DIR)
    elif opt.phase == "test" and opt.job_name.startswith("inf"):
        opt.job_name = opt.job_name.split('-', 1)[1]

        # Directories should already exist
        CACHE_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Cache/{opt.job_name}"
        FIG_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Figures/{opt.job_name}"
        LOG_DIR = f'/nfs/home/pedro/Outputs-MRA-GAN/Logs/{opt.job_name}'
        MODELS_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Models/{opt.job_name}"
    else:
        raise NameError("Job phase is test but job name does not start with inf!")

    # Re-assign checkpoints directory
    opt.checkpoints_dir = MODELS_DIR
    opt.log_dir = LOG_DIR

    # Other variables
    val_gap = 5
    running_iter = 0
    LOAD = True

    # Folds
    for fold in range(10):
        # Define models and associated variables
        print('\nFOLD', fold)
        # Pre-loading sequence: Two out channels correspond to number of classes
        # In channels: 1 for standard, 4 for coordconv!
        # betas = (0.5,
        #          0.999)  # Consider 0.5 for beta1! https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
        # betas_disc = (0.5,
        #               0.999)  # Consider 0.5 for beta1! https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
        # Generators
        if opt.final_act == "leaky":
            G_A_final_act = torch.nn.LeakyReLU()
        elif opt.final_act == "sigmoid":
            G_A_final_act = torch.nn.Sigmoid()
        G_A = nnUNet(1, 1, dropout_level=0, final_act=G_A_final_act)
        G_B = nnUNet(1, 1, dropout_level=0)

        # Discriminators
        D_A = NoisyMultiscaleDiscriminator3D(1, opt.ndf,
                                             opt.n_layers_D,
                                             nn.InstanceNorm3d, False, 1, False)

        D_B = NoisyMultiscaleDiscriminator3D(1, opt.ndf,
                                             opt.n_layers_D,
                                             nn.InstanceNorm3d, False, 1, False)

        # Associated variables
        G_A = nn.DataParallel(G_A)
        G_B = nn.DataParallel(G_B)
        D_A = nn.DataParallel(D_A)
        D_B = nn.DataParallel(D_B)

        # Optimizers + schedulers
        import itertools
        G_optimizer = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()),
                                       lr=opt.gen_lr, betas=(0.5, 0.999))
        D_optimizer = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                                       lr=opt.disc_lr, betas=(0.5, 0.999))
        # G_A_optimizer = torch.optim.Adam(G_A.parameters(), lr=0.0001, betas=betas)
        # G_B_optimizer = torch.optim.Adam(G_B.parameters(), lr=0.0001, betas=betas)
        # D_A_optimizer = torch.optim.Adam(D_A.parameters(), lr=0.000025, betas=betas_disc)
        # D_B_optimizer = torch.optim.Adam(D_B.parameters(), lr=0.000025, betas=betas_disc)

        # Domain A
        G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, 0.995)
        D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, 0.995)
        # G_A_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_A_optimizer, 0.99)
        # D_A_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_A_optimizer, 0.99)
        # G_A_scaler = torch.cuda.amp.GradScaler()
        # D_A_scaler = torch.cuda.amp.GradScaler()

        # Domain B
        # G_B_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_B_optimizer, 0.99)
        # D_B_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_B_optimizer, 0.99)
        # G_B_scaler = torch.cuda.amp.GradScaler()
        # D_B_scaler = torch.cuda.amp.GradScaler()

        # Push to device
        G_A.cuda()
        G_B.cuda()
        D_A.cuda()
        D_B.cuda()

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
        # model = create_model(opt)
        # model.setup(opt)

        # Model loading
        file_list = os.listdir(path=MODELS_DIR)
        num_files = len(file_list)
        print(f'The number of files is {num_files}')

        if num_files > 0 and LOAD:
            total_steps = model.load_networks('latest', models_dir=MODELS_DIR, phase=opt.phase)
        else:
            total_steps = 0
        visualizer = Visualizer(opt)

        # Z distribution
        z_sampler = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda:0")),
                                               torch.tensor(1.0).to(device=torch.device("cuda:0")))

        if opt.phase == "train":
            # Epochs
            for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
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

                    # Names (Not needed for now)
                    image_name = os.path.basename(train_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                    label_name = os.path.basename(train_sample[0]["label_meta_dict"]["filename_or_obj"][0])

                    # Pass inputs to model and optimise: Forward loop
                    fake_B = G_A(real_A)
                    # Pair fake B with fake z to generate rec_A
                    rec_A = G_B(fake_B)

                    # Backward loop: Sample z from normal distribution
                    fake_A = G_B(real_B)
                    # Reconstructed B
                    rec_B = G_A(fake_A)

                    # Identity
                    # idt_A = G_A(real_B)
                    # idt_B = G_B(real_A)

                    # Training
                    # Only begin to train discriminator after model has started to converge
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
                        # total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle + idt_A_loss + idt_B_loss
                        total_G_loss = G_A_loss + G_B_loss + A_cycle + B_cycle

                        # Backward
                        total_G_loss.backward()

                        # G optimization
                        G_optimizer.step()

                    # if total_steps % opt.print_freq == 0:
                    if total_steps % opt.save_latest_freq == 0:
                        print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                        # Saving
                        # Define ONE file for saving ALL state dicts
                        G_A.cpu()
                        G_B.cpu()
                        D_A.cpu()
                        D_B.cpu()
                        save_filename = f'epoch_{epoch}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
                        current_state_dict = {
                            'G_optimizer_state_dict': G_optimizer.state_dict(),
                            'D_optimizer_state_dict': D_optimizer.state_dict(),
                            'epoch': epoch,
                            'running_iter': running_iter,
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

                    if total_steps % 250 == 0:
                        # Graphs
                        writer.add_scalars('Loss/Adversarial',
                                           {"Total_G_loss": total_G_loss,
                                            "Generator_A": G_A_loss,
                                            "Generator_B": G_B_loss,
                                            "Discriminator_A": D_A_loss,
                                            "Discriminator_B": D_B_loss,
                                            }, running_iter)

                        writer.add_scalars('Loss/Granular_G',
                                           {"total_G_loss": total_G_loss,
                                            "Generator_A": G_A_loss,
                                            "Generator_B": G_B_loss,
                                            "cycle_A": A_cycle,
                                            "cycle_B": B_cycle,
                                            # "idt_A": idt_A,
                                            # "idt_B": idt_B
                                            }, running_iter)

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
                            image_name = os.path.basename(val_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                            label_name = os.path.basename(val_sample[0]["label_meta_dict"]["filename_or_obj"][0])
                            val_affine = val_sample[0]['image_meta_dict']['affine'][0, ...]
                            label_affine = val_sample[0]['label_meta_dict']['affine'][0, ...]

                            # Forward
                            # Pass inputs to model and optimise: Forward loop
                            val_fake_B = G_A(val_real_A)
                            # Pair fake B with fake z to generate rec_A
                            val_rec_A = G_B(val_fake_B)

                            # Backward loop
                            val_fake_A = G_B(val_real_B)
                            # Reconstructed B
                            val_rec_B = G_A(val_fake_A)

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

                            # Idt losses
                            # val_idt_A_loss = criterionIdt(val_idt_A, val_real_B)
                            # val_idt_B_loss = criterionIdt(val_idt_B, val_real_A)

                            # Total loss
                            # val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle + val_idt_A_loss + val_idt_B_loss
                            val_total_G_loss = val_G_A_loss + val_G_B_loss + val_A_cycle + val_B_cycle

                        # Graphs
                        writer.add_scalars('Loss/Val_Adversarial',
                                           {
                                            "Generator_A": val_G_A_loss,
                                            "Generator_B": val_G_B_loss,
                                            "Discriminator_A": val_D_A_loss,
                                            "Discriminator_B": val_D_B_loss,
                                           }, running_iter)

                        writer.add_scalars('Loss/Val_Granular_G',
                                           {"total_G_loss": val_total_G_loss,
                                            "Generator_A": val_G_A_loss,
                                            "Generator_B": val_G_B_loss,
                                            "cycle_A": val_A_cycle,
                                            "cycle_B": val_B_cycle,
                                            # "idt_A": val_idt_A,
                                            # "idt_B": val_idt_B
                                            }, running_iter)

                        # Saving
                        print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                        # Saving
                        # Define ONE file for saving ALL state dicts
                        G_A.cpu()
                        G_B.cpu()
                        D_A.cpu()
                        D_B.cpu()
                        save_filename = f'epoch_{epoch}_checkpoint_iters_{running_iter}_fold_{fold}.pth'
                        current_state_dict = {
                            'G_optimizer_state_dict': G_optimizer.state_dict(),
                            'D_optimizer_state_dict': D_optimizer.state_dict(),
                            'epoch': epoch,
                            'running_iter': running_iter,
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

                print(f'End of epoch {epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
                G_scheduler.step(epoch)
                D_scheduler.step(epoch)
        elif opt.phase == "test":
            # Carry out inference
            model.eval()
            with torch.no_grad():
                for inf_sample in inf_loader:
                    inf_image = inf_sample['image']
                    inf_label = inf_sample['label']
                    image_name = os.path.basename(inf_sample["image_meta_dict"]["filename_or_obj"][0])
                    label_name = os.path.basename(inf_sample["label_meta_dict"]["filename_or_obj"][0])
                    inf_affine = inf_sample['image_meta_dict']['affine'][0, ...]
                    label_affine = inf_sample['label_meta_dict']['affine'][0, ...]

                    model.set_input([inf_image, inf_label])
                    # model.optimize_parameters(training=False)
                    del inf_image, inf_label, inf_sample

                    # Inference
                    fake_B, rec_A, fake_A, rec_B = model.test_forward(overlap=0.0)

                    # Saving
                    def normalise_images(array):
                        import numpy as np
                        return (array - np.min(array)) / (np.max(array) - np.min(array))
                    
                    save_img(normalise_images(fake_B.cpu().detach().squeeze().numpy()),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_B_" + os.path.basename(image_name)))
                    # save_img(normalise_images(rec_A.cpu().detach().squeeze().numpy()),
                    #          inf_affine,
                    #          os.path.join(FIG_DIR, "Rec_A_" + os.path.basename(image_name)))
                    save_img(normalise_images(fake_A.cpu().detach().squeeze().numpy()),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_A_" + os.path.basename(label_name)))
                    # save_img(normalise_images(rec_B.cpu().detach().squeeze().numpy()),
                    #          inf_affine,
                    #          os.path.join(FIG_DIR, "Rec_B_" + os.path.basename(label_name)))
