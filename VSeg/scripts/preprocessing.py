import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scr.def4load import NM_DiArray, myload_DFKs_MAIN, myload_PrincDirects, VTF3D_RotMatrix4Vect, VTF3D_Rot3DwithR, VTF3D_rotate3DKernel
import monai
from monai.transforms import Compose, LoadNiftid, AddChanneld, Orientationd, NormalizeIntensityd, ToTensord, LoadNifti
from monai.data import list_data_collate
import glob
from torch.utils.data import DataLoader
from scr.prep_function import upsamp, savesum

#from fire import Fire
#import numpy.matlib
#import matplotlib
#import matplotlib.pyplot as plt
#import time
#import torchvision.transforms.functional as TF
#import nibabel as nib
#import datetime

class Const_strcData:
    pass

class preprocessing():
    def __init__(self, pwd, pp_choice, pp_inputfolder, pp_save_folder, pp_tmp_catch, pp_filetype, pp_para_folder):
        self.pwd = pwd
        self.choice = pp_choice
        self.inputfolder = pp_inputfolder
        self.pp_save_folder = pp_save_folder
        self.tmp_catch = pp_tmp_catch
        self.filetype = pp_filetype
        self.pp_para_folder = pp_para_folder

    def process(self):
        # Step 1: Create nessessary parameters for stafano's kernel
        #DFKs = myload_DFKs_MAIN(self.pwd + '/parameters/DFKs_MAIN.mat')
        DFKs = myload_DFKs_MAIN(self.pp_para_folder + '/DFKs_MAIN.mat')
        num_ker = 1  # only take the stick 12
        sel_k_no = 0  # selected kernel (this needed to be within the number of kernel definded by num_ker
        stickDFK = Const_strcData()
        stickDFK.Dk = [None] * num_ker  # or kernell
        stickDFK.phi = [None] * num_ker

        GRID = Const_strcData()
        GRID.X = [None] * num_ker
        GRID.Y = [None] * num_ker
        GRID.Z = [None] * num_ker
        GRID.pad = [None] * num_ker

        kernell = [None] * num_ker
        for i in range(num_ker):
            tmpDk = DFKs.Dk[0][i][:, :, :]  # [2:7,2:7,2:7]#np.zeros((5,5,5))
            tmpDk = torch.from_numpy(tmpDk)
            stickDFK.Dk[i] = tmpDk  # .to(device)

            tmpphi = DFKs.phi[0][i]
            tmpphi = torch.from_numpy(tmpphi)
            stickDFK.phi[i] = tmpphi  # .to(device)

            tmpX = np.array(DFKs.GRID_X[0][i])
            tmpY = np.array(DFKs.GRID_Y[0][i])
            tmpZ = np.array(DFKs.GRID_Z[0][i])
            tmppad = np.array(DFKs.GRID_pad[0][i])
            # tmpX = torch.from_numpy(tmpX)
            # tmpY = torch.from_numpy(tmpY)
            # tmpZ = torch.from_numpy(tmpZ)
            # tmppad = torch.from_numpy(tmppad)
            GRID.X[i] = tmpX  # .to(device)
            GRID.Y[i] = tmpY  # .to(device)
            GRID.Z[i] = tmpZ  # .to(device)
            GRID.pad[i] = tmppad  # .to(device)

        #PrincDirects = myload_PrincDirects(self.pwd + '/parameters/PrincDirects.mat')
        PrincDirects = myload_PrincDirects(self.pp_para_folder + '/PrincDirects.mat')
        ## Step 1.2: whi8ch is VT3D_rotatDFKStickAlongPrincDirects in MATLAB (line 857)
        GRID_use = Const_strcData()  # in MATLAB stefno used GRID (need to check with him)
        GRID_use.X = GRID.X[sel_k_no]
        GRID_use.Y = GRID.Y[sel_k_no]
        GRID_use.Z = GRID.Z[sel_k_no]
        GRID_use.pad = GRID.pad[sel_k_no]

        DFKOrientedSticks = Const_strcData()
        DFKOrientedSticks.Dk = [None] * len(PrincDirects)
        for ors in range(len(PrincDirects)):
            print(ors)
            ## Rotating the kernel Data according to the MainOrients
            Dkrot = VTF3D_rotate3DKernel(stickDFK.Dk[sel_k_no], stickDFK.phi[sel_k_no], PrincDirects[ors])  # Dkrot 9x9x9
            ## Cropping out the Padding
            ## Rotated Kerbel Tubularity Prob 'Dkrot'
            Dkrot = Dkrot[GRID_use.pad[0]:Dkrot.shape[0] - GRID_use.pad[0], GRID_use.pad[1]:Dkrot.shape[1] - GRID_use.pad[1], GRID_use.pad[2]:Dkrot.shape[2] - GRID_use.pad[2]]
            ## Balancing the kernel response
            Dkrot = Dkrot - (np.sum(Dkrot) / Dkrot.size)  # sum up to zero
            ## Assign and storing Data
            DFKOrientedSticks.Dk[ors] = Dkrot
            del Dkrot

        # Step 2: Convolution
        # Step 2.1: Rename DFKOrientedSticks.Dk into kernell
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kernell = [None] * len(DFKOrientedSticks.Dk)
        for i in range(len(DFKOrientedSticks.Dk)):
            tmp = DFKOrientedSticks.Dk[i]  # [2:7,2:7,2:7]#np.zeros((5,5,5))
            tmp = torch.from_numpy(tmp)
            kernell[i] = tmp.to(device)

        # Step 2.2: Load subject
        out_channels = 1
        in_channels = 1
        kh, kw, kt = 5, 5, 5
        down_fac = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        val_images = sorted(glob.glob(os.path.join(self.inputfolder, self.filetype)))
        val_files = [{'image': image_name} for image_name in zip(val_images)]
        val_files = val_files[:]
        val_transforms = Compose([
            LoadNiftid(keys=['image']),
            AddChanneld(keys=['image']),
            Orientationd(keys=['image'], axcodes='RAS'),
            NormalizeIntensityd(keys=['image'], channel_wise=True),
            ToTensord(keys=['image'])
        ])
        os.mkdir(self.tmp_catch)
        val_ds = monai.data.PersistentDataset(data=val_files, transform=val_transforms, cache_dir=self.tmp_catch)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=list_data_collate)

        OUTDIR = os.path.join(self.pwd, self.pp_save_folder)
        os.mkdir(OUTDIR)
        for s, val_data in enumerate(val_loader):
            loaderNII = LoadNifti(dtype=np.float32)
            load_file_name = val_files[s]['image']
            load_image_data, meta_load_image_data = loaderNII(load_file_name)
            affine_load = meta_load_image_data .get('affine')
            load_image_data = np.float32(load_image_data)
            load_image_data = load_image_data - load_image_data.min()  # normalisation
            load_image_data = load_image_data / load_image_data.max()  # normalisation

            image_data = np.zeros((out_channels, in_channels, load_image_data.shape[0], load_image_data.shape[1], load_image_data.shape[2]), dtype="Float32")  # (have 5 d incase we would like to have torch data structure
            image_data[0, 0, :, :, :] = load_image_data
            image_data_torch = torch.from_numpy(image_data)
            image_data_torch = image_data_torch.to(device)

            del image_data, load_image_data

            ## Unet downsampling
            # weight_use = [None]*len(DFKOrientedSticks.Dk)
            all_kernel_image = np.zeros((81, image_data_torch.shape[-3], image_data_torch.shape[-2], image_data_torch.shape[-1]))
            # all_kernel_image = all_kernel_image.to(device)
            for i in np.linspace(0, 80, 81).astype(dtype='int'):
                print('i=', i)
                weight = torch.randn(out_channels, in_channels, kh, kw, kt, requires_grad=True)
                weight = weight.to(device)
                with torch.no_grad():
                    weight[0, 0, :, :, :] = kernell[i]
                # weight_use[i] = weight
                # weight_use = weight

                ### Downsamle loop
                all_upDx = np.zeros((len(down_fac), image_data_torch.shape[-3], image_data_torch.shape[-2], image_data_torch.shape[-1]))
                # all_upDx = all_upDx.to(device)
                for j in range(len(down_fac)):
                    m_tmp = nn.AdaptiveAvgPool3d((int(down_fac[j] * image_data_torch.shape[-3]),
                                                  int(down_fac[j] * image_data_torch.shape[-2]),
                                                  int(down_fac[j] * image_data_torch.shape[-1])))
                    m_tmp = m_tmp.to(device)
                    # downDx_tmp = F.conv3d(m_tmp(image_data_torch), weight_use[i], stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1))
                    # downDx_tmp = F.conv3d(m_tmp(image_data_torch), weight_use, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1))
                    downDx_tmp = F.conv3d(m_tmp(image_data_torch).to(device), weight, stride=(1, 1, 1), padding=(2, 2, 2), dilation=(1, 1, 1))
                    # downDx_tmp[downDx_tmp < 0] = 0
                    # downDx_tmp = downDx_tmp * facc
                    upDx_tmp = upsamp(downDx_tmp, [image_data_torch.shape[-3], image_data_torch.shape[-2], image_data_torch.shape[-1]], device)
                    all_upDx[j, ...] = upDx_tmp.data.cpu().numpy()[0, 0, ...]
                    del m_tmp, downDx_tmp, upDx_tmp
                if self.choice == 'Avg':
                    tmp3 = np.mean(all_upDx, axis=0)
                elif self.choice == 'Max':
                    tmp3 = np.max(all_upDx, axis=0)
                elif self.choice == 'Sum':
                    tmp3 = np.sum(all_upDx, axis=0)
                all_kernel_image[i, ...] = tmp3
                del weight, tmp3, all_upDx
                ### End of downsample loop

            if self.choice == 'Avg':
                tmp4 = np.mean(all_kernel_image, axis=0)
                #savesum(tmp4, 'SLogAvg_' + str(s + 1) + self.filetype[1:], self.inputfolder, affine_load)
                savesum(tmp4, 'SLogAvg_' + str(s + 1) + self.filetype[1:], OUTDIR, affine_load)
            elif self.choice == 'Max':
                tmp4 = np.max(all_kernel_image, axis=0)
                #savesum(tmp4, 'SLogMax_' + str(s + 1) + self.filetype[1:], self.inputfolder, affine_load)
                savesum(tmp4, 'SLogMax_' + str(s + 1) + self.filetype[1:], OUTDIR, affine_load)
            elif self.choice == 'Sum':
                tmp4 = np.sum(all_kernel_image, axis=0)
                #savesum(tmp4, 'SLogSum_' + str(s + 1) + self.filetype[1:], self.inputfolder, affine_load)
                savesum(tmp4, 'SLogSum_' + str(s + 1) + self.filetype[1:], OUTDIR, affine_load)
            del all_kernel_image, tmp4
            del image_data_torch, affine_load, loaderNII
            del load_file_name

            print('finish one subject')

        x = 0
        return(x)


