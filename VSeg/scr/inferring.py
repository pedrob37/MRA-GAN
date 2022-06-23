import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import monai
from monai.transforms import AsDiscrete, Compose, LoadNiftid, AddChanneld, CropForegroundd, \
    Orientationd, ToTensord, NormalizeIntensityd, LoadNifti
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode
from monai.networks.layers import Norm
import nibabel as nib

# import matplotlib #chin
# import matplotlib.pyplot as plt
#import sys
#from torch import nn
#from monai.metrics import compute_meandice
# from monai.transforms import \
#     AsDiscrete, Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
#     RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, RandSpatialCrop, ToTensord, AsChannelFirstd, ScaleIntensityd, RandRotate90d, NormalizeIntensityd, \
#     LoadNifti

monai.config.print_config()

class Const_strcData:
    pass

class inferring():
    def __init__(self, pwd, ii_image, ii_image_k, ii_image_filetype, ii_image_k_filetype, ii_load_model_folder, ii_model_file, ii_save_folder, ii_ROI_val, ii_tmp_catch):
        self.pwd = pwd
        self.ii_image = ii_image
        self.ii_image_k = ii_image_k
        self.ii_image_filetype = ii_image_filetype
        self.ii_image_k_filetype = ii_image_k_filetype
        self.ii_load_model_folder = ii_load_model_folder
        self.ii_model_file = ii_model_file
        self.ii_save_folder = ii_save_folder
        self.ii_ROI_val = ii_ROI_val
        self.ii_tmp_catch = ii_tmp_catch

    def infer(self):
        print('torch.cuda.is_available()', torch.cuda.is_available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device=', device)

        train_images = sorted(glob.glob(os.path.join(self.ii_image, self.ii_image_filetype)))
        train_Kmax_imgs = sorted(glob.glob(os.path.join(self.ii_image_k, self.ii_image_k_filetype)))
        data_dicts = [{'image': image_name, 'Kmax': Kmax_name} for image_name, Kmax_name in zip(train_images, train_Kmax_imgs)]
        val_files = data_dicts[:]
        print('Load data')

        Num_in_ch = 2
        overlap = float(0.55)
        mode = BlendMode.GAUSSIAN

        OUTPUT_DIR = os.path.join(self.ii_load_model_folder, self.ii_save_folder)
        # self.pwd + '/' + self.pwd
        print('OUTPUT_DIR', OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)
        ROI_set = (self.ii_ROI_val, self.ii_ROI_val, self.ii_ROI_val)

        #val_interval = 2
        val_transforms = Compose([
            LoadNiftid(keys=['image', 'Kmax']),
            AddChanneld(keys=['image', 'Kmax']),
            Orientationd(keys=['image', 'Kmax'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image', 'Kmax'], channel_wise=True),
            CropForegroundd(keys=['image', 'Kmax'], source_key='image'),
            # Spacingd(keys=['image', 'label', 'Kmax'], pixdim=(new_spacing, new_spacing, new_spacing), interp_order=(3, 0), mode='nearest'),
            ToTensord(keys=['image', 'Kmax'])
        ])

        ## Define CacheDataset and DataLoader for training and validation
        print('self.ii_tmp_catch=', self.ii_tmp_catch)
        os.mkdir(self.ii_tmp_catch)
        # val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
        val_ds = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir=self.ii_tmp_catch)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)

        model = monai.networks.nets.UNet(dimensions=3, in_channels=Num_in_ch, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(1, 1, 1, 1), num_res_units=2, norm=Norm.BATCH).to(device)
        model.load_state_dict(torch.load(os.path.join(self.ii_load_model_folder, self.ii_model_file)))
        #facc = 1  # 40

        ## Inference
        model.eval()
        #model = model.to(device)

        post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
        post_label = AsDiscrete(to_onehot=True, n_classes=2)
        with torch.no_grad():
            mean_img = np.zeros((len(val_loader), 1))
            for i, val_data in enumerate(val_loader):
                roi_size = ROI_set
                sw_batch_size = 1
                ## load affine, it could be better solution than this but i don't know
                loader11 = LoadNifti(dtype=np.float32)
                img11, metadata11 = loader11(val_files[i]['image'])
                affine11 = metadata11.get('affine')
                del loader11, img11, metadata11
                ## end of load affine

                val_inputs, val_kmax_imgs = val_data['image'].to(device), val_data['Kmax'].to(device)
                #val_inputs, val_kmax_imgs = val_data['image'], val_data['Kmax']
                #val_inputs = val_inputs.to(device)
                #val_kmax_imgs = val_kmax_imgs.to(device)
                val_inputs2 = torch.zeros((val_inputs.shape[0], Num_in_ch, val_inputs.shape[2], val_inputs.shape[3], val_inputs.shape[4]))
                for ii in range(val_inputs.shape[0]):
                    val_inputs2[ii, 0, :, :, :] = val_inputs[ii, 0, :, :, :] #* facc
                    val_inputs2[ii, 1, :, :, :] = val_kmax_imgs[ii, 0, :, :, :] #* facc
                mean_img[i] = torch.mean(val_inputs2[ii, 0, :, :, :]).data.cpu().numpy()
                val_inputs2 = val_inputs2.to(device)

                # val_outputs = sliding_window_inference(val_inputs2, roi_size, sw_batch_size, model)
                val_outputs = sliding_window_inference(val_inputs2, roi_size, sw_batch_size, model, mode=mode, overlap=overlap)
                val_outputs = post_pred(val_outputs)
                # val_labels = post_label(val_labels)

                Imagee = np.squeeze(val_outputs.data.cpu().numpy()[0, 1, :, :, :]).astype(dtype='float32')
                # Imagee = np.squeeze(val_outputs2.data.cpu().numpy()[0, 0, :, :, :]).astype(dtype='float32')
                # new_image = nib.Nifti1Image(Imagee, affine=np.eye(4)) # did not use actual affine transformation
                new_image = nib.Nifti1Image(Imagee, affine=affine11)
                del affine11

                NIIname = 'SPredict_' + str(i + 1) + '.nii.gz'
                nib.save(new_image, os.path.join(OUTPUT_DIR, NIIname))
                del val_outputs, Imagee, new_image, NIIname

        x = 0
        return x


