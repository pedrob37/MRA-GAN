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
from test import inference
import monai
from monai.transforms import Compose, LoadNiftid, AddChanneld, CropForegroundd, RandCropByPosNegLabeld, Orientationd, ToTensord, NormalizeIntensityd, RandWeightedCropd
from monai.data import list_data_collate#, sliding_window_inference
import datetime
import nibabel as nib
class Const_strcData:
    pass

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()

    ### Chin-MONAI
    para = Const_strcData()
    para.n_samp = 2
    para.patch_sz0 = opt.patch_size[0]
    para.patch_sz1 = opt.patch_size[1]
    para.patch_sz2 = opt.patch_size[2]
    train_images = sorted(glob.glob(os.path.join(opt.data_path, 'images', '*.nii')))
    train_labels = sorted(glob.glob(os.path.join(opt.data_path, 'labels', '*.nii')))
    train_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files = train_files[:]
    train_transforms = Compose([LoadNiftid(keys=['image', 'label']),
                                AddChanneld(keys=['image', 'label']),
                                Orientationd(keys=['image', 'label'], axcodes='RAS'),
                                NormalizeIntensityd(keys=['image'], channel_wise=True),
                                # RandGaussianNoised(keys=['image'], prob=0.75, mean=0.0, std=1.75),
                                # RandRotate90d(keys=['image', 'heatmap', 'paf'], prob=0.5, spatial_axes=[0, 2]),
                                CropForegroundd(keys=['image', 'label'], source_key='image'),
                                RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=[opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]], pos=20, neg=0, num_samples=para.n_samp, image_threshold=-1),
                                # RandWeightedCropd(keys=['image', 'label'], w_key='label', spatial_size=[opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]], num_samples=para.n_samp),
                                # Spacingd(keys=['image', 'label'], pixdim=(opt.new_resolution[0], opt.new_resolution[0], opt.new_resolution[0]), interp_order=(3, 0), mode='nearest'),
                                ToTensord(keys=['image', 'label'])])

    ## Define CacheDataset and DataLoader for training and validation
    tmp_catch = '/media/chayanin/Storage/chin/catch' + '/' + str(datetime.datetime.now())
    os.mkdir(tmp_catch)
    train_ds = monai.data.PersistentDataset(data=train_files, transform=train_transforms, cache_dir=tmp_catch)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)#, cache_dir=tmp_catch)
    train_loader_MONAI = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, collate_fn=list_data_collate)

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

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # chin added 20220128
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

        for i, patch_s in enumerate(train_loader_MONAI):
        #for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            dataMONAI = []
            dataMONAI.append([])
            dataMONAI.append([])
            dataMONAI[0] = patch_s['image']
            dataMONAI[1] = patch_s['label']
            #model.set_input(data)
            model.set_input(dataMONAI)
            model.optimize_parameters()
            del dataMONAI

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()










