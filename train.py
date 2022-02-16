import os
import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
from utils.utils import create_path, save_img
# from test import inference
import monai
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
import datetime
import nibabel as nib


# class Const_strcData:
#     pass

if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    ### MONAI
    train_images = sorted(glob.glob(os.path.join(opt.data_path, 'Images', '*.nii*')))
    train_labels = sorted(glob.glob(os.path.join(opt.data_path, 'Labels', '*.nii*')))
    
    val_images = sorted(glob.glob(os.path.join(opt.data_path, 'Images', '*.nii*')))
    val_labels = sorted(glob.glob(os.path.join(opt.data_path, 'Labels', '*.nii*')))

    # Isolate and remove non-matching images
    train_images = [x for x in train_images if os.path.join(opt.data_path, 'Labels', os.path.basename(x)[:-8] + "2.nii.gz") in train_labels]
    train_labels = [x for x in train_labels if os.path.join(opt.data_path, 'Images', os.path.basename(x)[:-8] + "1.nii.gz") in train_images]

    # Remove bad images
    train_images = [x for x in train_images if "IXI014-HH-1236" not in x]
    train_labels = [x for x in train_labels if "IXI014-HH-1236" not in x]

    train_files = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(train_images, train_labels)]

    val_files = [{'image': image_name, 'label': label_name}
                 for image_name, label_name in zip(val_images, val_labels)]

    assert len(train_images) == len(train_labels)
    train_transforms = Compose([LoadImaged(keys=['image', 'label']),
                                AddChanneld(keys=['image', 'label']),
                                # Orientationd(keys=['image', 'label'], axcodes='RAS'),
                                RandSpatialCropSamplesd(keys=["image", "label"],
                                                        roi_size=opt.patch_size,
                                                        random_center=True,
                                                        random_size=False,
                                                        num_samples=1),
                                SpatialPadd(keys=["image", "label"],
                                            spatial_size=opt.patch_size),
                                NormalizeIntensityd(keys=['image'], channel_wise=True),
                                ToTensord(keys=['image', 'label'])])

    val_transforms = Compose([LoadImaged(keys=['image', 'label']),
                              AddChanneld(keys=['image', 'label']),
                              # Orientationd(keys=['image', 'label'], axcodes='RAS'),
                              RandSpatialCropSamplesd(keys=["image", "label"],
                                                      roi_size=opt.patch_size,
                                                      random_center=True,
                                                      random_size=False,
                                                      num_samples=1),
                              SpatialPadd(keys=["image", "label"],
                                          spatial_size=opt.patch_size),
                              NormalizeIntensityd(keys=['image'], channel_wise=True),
                              ToTensord(keys=['image', 'label'])])

    ## Relevant directories
    CACHE_DIR = "/home/pedro/MRA-GAN/MRA-GAN/Cache"
    FIG_DIR = "/home/pedro/MRA-GAN/MRA-GAN/Figures"
    create_path(CACHE_DIR)
    create_path(FIG_DIR)

    # Other variables
    val_gap = 5

    # Training + validation
    train_ds = monai.data.PersistentDataset(data=train_files,
                                            transform=train_transforms,
                                            cache_dir=CACHE_DIR
                                            )

    train_loader = DataLoader(dataset=train_ds,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.workers,
                              )

    val_ds = monai.data.PersistentDataset(data=val_files,
                                          transform=val_transforms,
                                          cache_dir=CACHE_DIR
                                          )

    val_loader = DataLoader(dataset=val_ds,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.workers,
                            )

    ### End of Chin-MONAI

    # # -----  Transformation and Augmentation process for the data  -----
    # min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    # trainTransforms = [
    #             NiftiDataset.Resample(opt.new_resolution, opt.resample),
    #             NiftiDataset.Augmentation(),
    #             NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
    #             NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    #             ]
    #
    # train_set = NifitDataSet(opt.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=True, train=True)
    # print('lenght train list:', len(train_set))
    # train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size
    # # -----------------------------------------------------

    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # save_fold_tmp = os.getcwd() + '/tmp_patch'
        # os.mkdir(save_fold_tmp)
        # for i, data in enumerate(train_loader):
        #     #data2 = data
        #     print('chin')
        #     data0 = np.float32(np.squeeze(data[0]))
        #     data1 = np.float32(np.squeeze(data[1]))
        #     save_img0 = nib.Nifti1Image(data0, affine=np.eye(4))
        #     save_img1 = nib.Nifti1Image(data1, affine=np.eye(4))
        #     nib.save(save_img0, save_fold_tmp + '/save_img0_i=' + str(i) + '.nii.gz')
        #     nib.save(save_img1, save_fold_tmp + '/save_img1_i=' + str(i) + '.nii.gz')
        #     del data0, data1, save_img0, save_img1

        # save_fold_tmp_MONAI = os.getcwd() + '/tmp_patch_MONAI'
        # os.mkdir(save_fold_tmp_MONAI)
        # for i, patch_s in enumerate(train_loader_MONAI):
        #     print('chin-MONAI')
        #     for j in range(patch_s['image'].data.cpu().numpy().shape[0]):
        #         data0 = np.float32(np.squeeze(patch_s['image'][j, 0, ...].data.cpu().numpy()))
        #         data1 = np.float32(np.squeeze(patch_s['label'][j, 0, ...].data.cpu().numpy()))
        #         save_img0 = nib.Nifti1Image(data0, affine=np.eye(4))
        #         save_img1 = nib.Nifti1Image(data1, affine=np.eye(4))
        #         nib.save(save_img0, save_fold_tmp_MONAI + '/save_img0_i=' + str(i) + 'j=' + str(j) +'.nii.gz')
        #         nib.save(save_img1, save_fold_tmp_MONAI + '/save_img1_i=' + str(i) + 'j=' + str(j) +'.nii.gz')
        #         del data0, data1, save_img0, save_img1
        # Training
        for i, train_sample in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()

            # Training variables
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            train_image = train_sample[0]['image']
            train_label = train_sample[0]['label']
            image_name = os.path.basename(train_sample[0]["image_meta_dict"]["filename_or_obj"][0])
            label_name = os.path.basename(train_sample[0]["label_meta_dict"]["filename_or_obj"][0])
            train_affine = train_sample[0]['image_meta_dict']['affine'][0, ...]
            label_affine = train_sample[0]['label_meta_dict']['affine'][0, ...]

            print(f"Input shapes: {train_image.shape}, {train_label.shape}")
            # save_img(train_image.cpu().detach().squeeze().numpy(), train_affine, os.path.join(FIG_DIR, image_name))
            # save_img(train_label.cpu().detach().squeeze().numpy(), label_affine, os.path.join(FIG_DIR, label_name))

            model.set_input([train_image, train_label])
            model.optimize_parameters()
            del train_image, train_label

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, total_steps {total_steps})')
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {total_steps}')
            model.save_networks('latest')
            model.save_networks(epoch)

        if val_gap % 5 == 0:
            for val_sample in val_loader:
                # Complete this


        print(f'End of epoch {epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
        model.update_learning_rate()
