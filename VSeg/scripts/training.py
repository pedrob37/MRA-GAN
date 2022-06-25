import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import monai
from monai.transforms import AsDiscrete, Compose, LoadNiftid, AddChanneld, CropForegroundd, \
    RandCropByPosNegLabeld, Orientationd, ToTensord, NormalizeIntensityd
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from torch.utils.tensorboard import SummaryWriter

#import sys
#import matplotlib #chin
#import matplotlib.pyplot as plt
# from monai.transforms import \
#     AsDiscrete, Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
#     RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, RandSpatialCrop, ToTensord, AsChannelFirstd, ScaleIntensityd, RandRotate90d, NormalizeIntensityd

class training():
    def __init__(self, pwd, tt_image, tt_label, tt_image_k, tt_image_filetype, tt_label_filetype, tt_image_k_filetype, tt_tmp_catch_train, tt_tmp_catch_val, tt_numval, tt_Nepoch, tt_patch_size, tt_ROI_val, tt_num_samples, tt_mainsavefold):
        self.pwd = pwd
        self.tt_image = tt_image
        self.tt_label = tt_label
        self.tt_image_k = tt_image_k
        self.tt_image_filetype = tt_image_filetype
        self.tt_label_filetype = tt_label_filetype
        self.tt_image_k_filetype = tt_image_k_filetype
        self.tt_tmp_catch_train = tt_tmp_catch_train
        self.tt_tmp_catch_val = tt_tmp_catch_val
        self.tt_numval = tt_numval
        self.tt_Nepoch = tt_Nepoch
        self.tt_patch_size = tt_patch_size
        self.tt_ROI_val = tt_ROI_val
        self.tt_num_samples = tt_num_samples
        self.tt_mainsavefold = tt_mainsavefold

    def train(self):
        train_images = sorted(glob.glob(os.path.join(self.tt_image, self.tt_image_filetype)))
        train_labels = sorted(glob.glob(os.path.join(self.tt_label, self.tt_label_filetype)))
        train_Kmax_imgs = sorted(glob.glob(os.path.join(self.tt_image_k, self.tt_image_k_filetype)))
        print(train_images[0])
        print(train_labels[0])
        print(train_Kmax_imgs[0])
        data_dicts = [{'image': image_name, 'label': label_name, 'Kmax': Kmax_name}
                      for image_name, label_name, Kmax_name in zip(train_images, train_labels, train_Kmax_imgs)]
        train_files, val_files = data_dicts[:-self.tt_numval], data_dicts[-self.tt_numval:]

        ## Step 2: MONAI preprocessing
        #new_spacing = 1.0
        self.tt_patch_size = 50
        Num_in_ch = 2
        train_transforms = Compose([
            LoadNiftid(keys=['image', 'label', 'Kmax']),
            AddChanneld(keys=['image', 'label', 'Kmax']),
            Orientationd(keys=['image', 'label', 'Kmax'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image', 'Kmax'], channel_wise=True),
            CropForegroundd(keys=['image', 'label', 'Kmax'], source_key='image'),
            RandCropByPosNegLabeld(
                keys=['image', 'label', 'Kmax'], label_key='label', spatial_size=[self.tt_patch_size, self.tt_patch_size, self.tt_patch_size],
                pos=1, neg=1, num_samples=self.tt_num_samples
            ),
            # Spacingd(keys=['image', 'label', 'Kmax'], pixdim=(new_spacing, new_spacing, new_spacing), interp_order=(3, 0), mode='nearest'),
            ToTensord(keys=['image', 'label', 'Kmax'])
        ])

        val_transforms = Compose([
            LoadNiftid(keys=['image', 'label', 'Kmax']),
            AddChanneld(keys=['image', 'label', 'Kmax']),
            Orientationd(keys=['image', 'label', 'Kmax'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image', 'Kmax'], channel_wise=True),
            CropForegroundd(keys=['image', 'label', 'Kmax'], source_key='image'),
            # Spacingd(keys=['image', 'label', 'Kmax'], pixdim=(new_spacing, new_spacing, new_spacing), interp_order=(3, 0), mode='nearest'),
            ToTensord(keys=['image', 'label', 'Kmax'])
        ])

        # Set deterministic training for reproducibility
        train_transforms.set_random_state(seed=0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.mkdir(self.tt_tmp_catch_train)
        # train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_ds = monai.data.PersistentDataset(data=train_files, transform=train_transforms, cache_dir=self.tt_tmp_catch_train)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1, collate_fn=list_data_collate)

        os.mkdir(self.tt_tmp_catch_val)
        # val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
        val_ds = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir=self.tt_tmp_catch_val)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=list_data_collate)

        ## Step 3: Create Model, Loss, Optimizer
        # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
        #device = torch.device('cuda:0')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = monai.networks.nets.UNet(dimensions=3, in_channels=Num_in_ch, out_channels=2,
                                         channels=(16, 32, 64, 128, 256),
                                         strides=(1, 1, 1, 1), num_res_units=2, norm=Norm.BATCH).to(device)
        loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)

        ## Step 4: Execute a typical PyTorch training process
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()

        #save_folder = self.pwd + '/Epoch_' + str(self.tt_Nepoch)
        save_folder = self.tt_mainsavefold + '/Epoch_' + str(self.tt_Nepoch)
        if os.path.exists(save_folder):
            print("save path is already existed.")
        else:
            os.mkdir(save_folder)

        # overlap = float(0.55)
        # mode = BlendMode.GAUSSIAN

        ROI_set = (self.tt_ROI_val, self.tt_ROI_val, self.tt_ROI_val)
        writerUNet = SummaryWriter(save_folder + '/runUNet')
        for epoch in range(self.tt_Nepoch):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch + 1, self.tt_Nepoch))
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels, kmax_imgs = batch_data['image'].to(device), batch_data['label'].to(device), batch_data[
                    'Kmax'].to(device)

                inputs2 = torch.zeros((inputs.shape[0], Num_in_ch, inputs.shape[2], inputs.shape[3], inputs.shape[4]))
                for ii in range(inputs.shape[0]):
                    inputs2[ii, 0, :, :, :] = inputs[ii, 0, :, :, :]
                    inputs2[ii, 1, :, :, :] = kmax_imgs[ii, 0, :, :, :]
                inputs2 = inputs2.to(device)
                optimizer.zero_grad()
                outputs = model(inputs2)
                loss = loss_function(outputs, labels)
                writerUNet.add_scalar("Loss", loss, epoch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print('{}/{}, train_loss: {:.4f}'.format(step, len(train_ds) // train_loader.batch_size, loss.item()))
                writerUNet.flush()

                del inputs, labels, kmax_imgs
                del inputs2
                del batch_data
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))
            torch.save(model.state_dict(), save_folder + '/' + 'last_model' + '_Nep' + str(self.tt_Nepoch) + '.pth')  # chin 2021.04.25

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    metric_sum = 0.
                    metric_count = 0
                    for val_data in val_loader:
                        val_inputs, val_labels, val_kmax_imgs = val_data['image'].to(device), val_data['label'].to(device), val_data['Kmax'].to(device)
                        val_inputs2 = torch.zeros((val_inputs.shape[0], Num_in_ch, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4]))
                        for ii in range(val_inputs.shape[0]):
                            val_inputs2[ii, 0, :, :, :] = val_inputs[ii, 0, :, :, :]
                            val_inputs2[ii, 1, :, :, :] = val_kmax_imgs[ii, 0, :, :, :]

                        val_inputs2 = val_inputs2.to(device)
                        roi_size = ROI_set
                        sw_batch_size = 1
                        val_outputs2 = sliding_window_inference(val_inputs2, roi_size, sw_batch_size, model)
                        # val_outputs2 = sliding_window_inference(val_inputs2, roi_size, sw_batch_size, model, mode=mode, overlap=overlap)
                        value = compute_meandice(y_pred=val_outputs2, y=val_labels, include_background=False, to_onehot_y=True, mutually_exclusive=True)
                        metric_count += len(value)
                        metric_sum += value.sum().item()
                        del val_inputs, val_labels, val_kmax_imgs
                        del val_inputs2
                        del val_data
                    metric = metric_sum / metric_count
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), save_folder + '/' + 'GD_best_metric_model' + '_Nep' + str(self.tt_Nepoch) + '.pth')
                        print('saved new best metric model')
                    print('current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(epoch + 1, metric, best_metric, best_metric_epoch))

        print('train completed, best_metric: {:.4f}  at epoch: {}'.format(best_metric, best_metric_epoch))

        np.save(save_folder + '/' + 'epoch_loss_values.npy', epoch_loss_values)
        np.save(save_folder + '/' + 'metric_values.npy', metric_values)

        x = 0
        return x