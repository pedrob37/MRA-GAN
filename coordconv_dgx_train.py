import os
import torch
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
import time
from models import create_model
from utils.visualizer import Visualizer
from utils.utils import create_path, save_img, CoordConvd
import monai
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
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

    # Augmentations
    # Transforms
    if opt.phase == "train":
        if opt.augmentation_level == "heavy":
            train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                        RandAffined(keys=["image", "label"],
                                                    scale_range=(0.1, 0.1, 0.1),
                                                    rotate_range=(0.25, 0.25, 0.25),
                                                    translate_range=(20, 20, 20),
                                                    mode=("nearest", "bilinear"),
                                                    as_tensor_output=False, prob=1.0,
                                                    padding_mode=('zeros', 'zeros')),
                                        RandGaussianSmoothd(keys=["label"], prob=0.25,  # 0.2
                                                            sigma_x=(0.25, 0.3),
                                                            sigma_y=(0.25, 0.3),
                                                            sigma_z=(0.25, 0.3)),
                                        RandBiasFieldd(keys=["label"], degree=3, coeff_range=(0.1, 0.25),
                                                       prob=0.25),  # Odd behaviour...
                                        NormalizeIntensityd(keys=['label'], channel_wise=True),
                                        RandGaussianNoiseD(keys=["label"], std=0.2, prob=0.5),
                                        RandSpatialCropSamplesd(keys=["image", "label"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label'])])
        elif opt.augmentation_level == "light":
            train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                        RandAffined(keys=["image", "label", "coords"],
                                                    scale_range=(0.1, 0.1, 0.1),
                                                    rotate_range=(0.25, 0.25, 0.25),
                                                    translate_range=(20, 20, 20),
                                                    mode=("nearest", "bilinear", "nearest"),
                                                    as_tensor_output=False, prob=1.0,
                                                    padding_mode=('zeros', 'zeros', 'border')),
                                        NormalizeIntensityd(keys=['label'], channel_wise=True),
                                        RandSpatialCropSamplesd(keys=["image", "label", "coords"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label'])])
        elif opt.augmentation_level == "none":
            train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                        AddChanneld(keys=['image', 'label']),
                                        CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                        NormalizeIntensityd(keys=['label'], channel_wise=True),
                                        RandSpatialCropSamplesd(keys=["image", "label"],
                                                                roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                                random_center=True,
                                                                random_size=False,
                                                                num_samples=1),
                                        ToTensord(keys=['image', 'label'])])

        val_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                  AddChanneld(keys=['image', 'label']),
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                  NormalizeIntensityd(keys=['label'], channel_wise=True),
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
                                  CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                  # RandSpatialCropSamplesd(keys=["image", "label"],
                                  #                         roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                  #                         random_center=True,
                                  #                         random_size=False,
                                  #                         num_samples=1),
                                  # SpatialPadd(keys=["image", "label"],
                                  #             spatial_size=opt.patch_size),
                                  NormalizeIntensityd(keys=['label'], channel_wise=True),
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
    LOAD = True

    # Folds
    for fold in range(10):
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
        model = create_model(opt)
        model.setup(opt)

        # Model loading
        file_list = os.listdir(path=MODELS_DIR)
        num_files = len(file_list)
        print(f'The number of files is {num_files}')

        if num_files > 0 and LOAD:
            total_steps = model.load_networks('latest', models_dir=MODELS_DIR, phase=opt.phase)
        else:
            total_steps = 0
        visualizer = Visualizer(opt)

        if opt.phase == "train":
            # Epochs
            for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
                # Model to train mode after potential eval call when running validation
                model.train()
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0

                # Iterations
                for i, train_sample in enumerate(train_loader):
                    iter_start_time = time.time()
                    if total_steps % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    visualizer.reset()

                    # Training variables: Global
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size

                    # Iteration-specific data
                    train_image = train_sample[0]['image'].cuda(non_blocking=True)
                    train_label = train_sample[0]['label'].cuda(non_blocking=True)
                    train_coords = train_sample[0]['coords'].cuda(non_blocking=True)

                    # Concatenate coordinates to channel dimension
                    print(train_image.shape, train_coords.shape)
                    train_image = torch.cat((train_image, train_coords), dim=1)
                    train_label = torch.cat((train_label, train_coords), dim=1)
                    print(train_image.shape)

                    # Names (Not needed for now)
                    image_name = os.path.basename(train_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                    label_name = os.path.basename(train_sample[0]["label_meta_dict"]["filename_or_obj"][0])

                    # Pass inputs to model and optimise
                    model.set_input([train_image, train_label, train_coords])
                    model.optimize_parameters(training=True)
                    del train_image, train_label, train_sample, train_coords

                    if total_steps % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        t = (time.time() - iter_start_time) / opt.batch_size
                        try:
                            visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                        except:
                            print("Passing for now")

                    if total_steps % opt.save_latest_freq == 0:
                        losses = model.get_current_losses()
                        print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                        model.save_networks(epoch, current_iter=total_steps, models_dir=MODELS_DIR)

                    if total_steps % 250 == 0:
                        model.write_logs(training=True,
                                         step=total_steps,
                                         current_writer=writer)
                        model.write_images(training=True,
                                           step=total_steps,
                                           current_writer=writer,
                                           current_opt=opt,
                                           current_fold=fold)
                    iter_data_time = time.time()

                if epoch % opt.save_epoch_freq == 0:
                    # model.save_networks('latest')
                    model.save_networks(epoch, current_iter=total_steps, models_dir=MODELS_DIR)

                if epoch % val_gap == 0:
                    model.eval()
                    with torch.no_grad():
                        for val_sample in val_loader:
                            # Validation variables
                            val_image = val_sample[0]['image'].cuda(non_blocking=True)
                            val_label = val_sample[0]['label'].cuda(non_blocking=True)
                            val_coords = val_sample[0]['coords'].cuda(non_blocking=True)

                            # Concatenate coordinates to channel dimension
                            print(val_image.shape, val_coords.shape)
                            val_image = torch.cat((val_image, val_coords), dim=1)
                            val_label = torch.cat((val_label, val_coords), dim=1)
                            print(val_image.shape)

                            image_name = os.path.basename(val_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                            label_name = os.path.basename(val_sample[0]["label_meta_dict"]["filename_or_obj"][0])
                            val_affine = val_sample[0]['image_meta_dict']['affine'][0, ...]
                            label_affine = val_sample[0]['label_meta_dict']['affine'][0, ...]

                            model.set_input([val_image, val_label, val_coords])
                            model.optimize_parameters(training=False)
                            del val_image, val_label, val_sample, val_coords

                            # if total_steps % opt.print_freq == 0:
                            #     losses = model.get_current_losses()
                            #     t = (time.time() - iter_start_time) / opt.batch_size
                            #     visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                            if total_steps % opt.save_latest_freq == 0:
                                print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                                model.save_networks('latest')

                        # Log validation performance
                        model.write_logs(training=False,
                                         step=total_steps,
                                         current_writer=writer)

                print(f'End of epoch {epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
                model.update_learning_rate()
        elif opt.phase == "test":
            # Carry out inference
            model.eval()
            with torch.no_grad():
                for inf_sample in inf_loader:
                    inf_image = inf_sample['image']
                    inf_label = inf_sample['label']
                    inf_coords = inf_sample[0]['coords'].cuda(non_blocking=True)

                    # Concatenate coordinates to channel dimension
                    print(inf_image.shape, inf_coords.shape)
                    inf_image = torch.cat((inf_image, inf_coords), dim=1)
                    inf_label = torch.cat((inf_label, inf_coords), dim=1)
                    print(inf_image.shape)

                    image_name = os.path.basename(inf_sample["image_meta_dict"]["filename_or_obj"][0])
                    label_name = os.path.basename(inf_sample["label_meta_dict"]["filename_or_obj"][0])
                    inf_affine = inf_sample['image_meta_dict']['affine'][0, ...]
                    label_affine = inf_sample['label_meta_dict']['affine'][0, ...]

                    model.set_input([inf_image, inf_label, inf_coords])
                    # model.optimize_parameters(training=False)
                    del inf_image, inf_label, inf_sample, inf_coords

                    # Inference
                    fake_B, rec_A, fake_A, rec_B = model.test_forward(overlap=0.3)
                    assert inf_affine == label_affine
                    print(fake_B.shape)
                    # Saving
                    save_img(fake_B.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_B_" + os.path.basename(image_name)))
                    save_img(rec_A.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Rec_A_" + os.path.basename(image_name)))
                    save_img(fake_A.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Fake_A_" + os.path.basename(label_name)))
                    save_img(rec_B.cpu().detach().squeeze().numpy(),
                             inf_affine,
                             os.path.join(FIG_DIR, "Rec_B_" + os.path.basename(label_name)))
