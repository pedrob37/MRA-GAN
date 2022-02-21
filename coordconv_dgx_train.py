import os
import sys
from utils.NiftiDataset import *
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
                              CropForegroundd,
                              RandCropByPosNegLabeld,
                              Orientationd,
                              ToTensord,
                              RandSpatialCropSamplesd,
                              NormalizeIntensityd,
                              RandWeightedCropd,
                              SpatialCropd,
                              SpatialPadd,
                              )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    ### MONAI
    # Data directory
    images_dir = os.path.join(opt.data_path, 'Images')
    labels_dir = os.path.join(opt.data_path, 'Labels')

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

    # Augmentations
    # Transforms
    train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                AddChanneld(keys=['image', 'label']),
                                CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),  # (1, 2, 3)),
                                RandSpatialCropSamplesd(keys=["image", "label", "coords"],
                                                        roi_size=(opt.patch_size, opt.patch_size, opt.patch_size),
                                                        random_center=True,
                                                        random_size=False,
                                                        num_samples=1),
                                # SpatialPadd(keys=["image", "label"],
                                #             spatial_size=opt.patch_size),
                                NormalizeIntensityd(keys=['image', "label", "coords"], channel_wise=True),
                                ToTensord(keys=['image', 'label', "coords"])])

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
                              NormalizeIntensityd(keys=['image', "label", "coords"], channel_wise=True),
                              ToTensord(keys=['image', 'label', "coords"])])

    ## Relevant job directories
    CACHE_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Cache/{opt.job_name}"
    FIG_DIR = f"/nfs/home/pedro/Outputs-MRA-GAN/Figures/{opt.job_name}"
    LOG_DIR = f'/nfs/home/pedro/Outputs-MRA-GAN/Logs/{opt.job_name}'
    create_path(CACHE_DIR)
    create_path(FIG_DIR)
    create_path(LOG_DIR)

    # Other variables
    val_gap = 5

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

        # Basic length checks
        assert len(train_df.Filename) == len(train_df.Label_Filename)
        assert len(val_df.Filename) == len(val_df.Label_Filename)

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

        # Model creation
        model = create_model(opt)
        model.setup(opt)
        if opt.epoch_count > 1:
            model.load_networks(opt.epoch_count)
        visualizer = Visualizer(opt)
        total_steps = 0

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
                # CoordConv
                train_coords = train_sample[0]['coords'].cuda(non_blocking=True)

                # Concatenate coordinates to channel dimension
                print(train_image.shape, train_coords.shape)
                train_image = torch.cat((train_image, train_coords), dim=1)
                print(train_image.shape)

                # Names (Not needed for now)
                image_name = os.path.basename(train_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                label_name = os.path.basename(train_sample[0]["label_meta_dict"]["filename_or_obj"][0])

                # Pass inputs to model and optimise
                model.set_input([train_image, train_label])
                model.optimize_parameters(training=True)
                del train_image, train_label, train_coords, train_sample

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    # print(f"Losses are {losses}")

                if total_steps % opt.save_latest_freq == 0:
                    losses = model.get_current_losses()
                    print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                    model.save_networks(epoch, current_iter=total_steps)

                if total_steps % 250 == 0:
                    model.write_logs(training=True,
                                     step=total_steps,
                                     current_writer=writer)
                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:
                # model.save_networks('latest')
                model.save_networks(epoch, current_iter=total_steps)

            if epoch % val_gap == 0:
                model.eval()
                with torch.no_grad():
                    for val_sample in val_loader:
                        # Complete this
                        # Validation variables
                        val_image = val_sample[0]['image'].cuda(non_blocking=True)
                        val_label = val_sample[0]['label'].cuda(non_blocking=True)
                        # CoordConv
                        val_coords = val_sample[0]['coords'].cuda(non_blocking=True)

                        image_name = os.path.basename(val_sample[0]["image_meta_dict"]["filename_or_obj"][0])
                        label_name = os.path.basename(val_sample[0]["label_meta_dict"]["filename_or_obj"][0])
                        val_affine = val_sample[0]['image_meta_dict']['affine'][0, ...]
                        label_affine = val_sample[0]['label_meta_dict']['affine'][0, ...]

                        model.set_input([val_image, val_label])
                        model.optimize_parameters(training=False)
                        del val_image, val_label, val_coords, val_sample

                        if total_steps % opt.print_freq == 0:
                            losses = model.get_current_losses()
                            t = (time.time() - iter_start_time) / opt.batch_size
                            visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

                        if total_steps % opt.save_latest_freq == 0:
                            print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                            model.save_networks('latest')

                    # Log validation performance
                    model.write_logs(training=False,
                                     step=total_steps,
                                     current_writer=writer)

            print(f'End of epoch {epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
            model.update_learning_rate()