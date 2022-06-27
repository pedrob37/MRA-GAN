import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from VSeg.scripts.def4load import myload_DFKs_MAIN, myload_PrincDirects, VTF3D_rotate3DKernel
from VSeg.scripts.prep_function import upsamp
from utils.utils import *


class Const_strcData:
    pass


class preprocessing():
    def __init__(self, para_folder, image_data):
        self.para_folder = para_folder
        self.image_data = image_data

    def process(self):
        # Step 1: Create necessary parameters
        DFKs = myload_DFKs_MAIN(self.para_folder + '/DFKs_MAIN.mat')
        num_ker = 1  # only take the stick 12
        sel_k_no = 0  # selected kernel (this needed to be within the number of kernel definded by num_ker
        stickDFK = Const_strcData()
        stickDFK.Dk = [None] * num_ker
        stickDFK.phi = [None] * num_ker

        GRID = Const_strcData()
        GRID.X = [None] * num_ker
        GRID.Y = [None] * num_ker
        GRID.Z = [None] * num_ker
        GRID.pad = [None] * num_ker

        for i in range(num_ker):
            stickDFK.Dk[i] = torch.FloatTensor(DFKs.Dk[0][i]).cuda()

            stickDFK.phi[i] = torch.FloatTensor(DFKs.phi[0][i]).cuda()

            GRID.X[i] = torch.FloatTensor(DFKs.GRID_X[0][i]).cuda()
            GRID.Y[i] = torch.FloatTensor(DFKs.GRID_Y[0][i]).cuda()
            GRID.Z[i] = torch.FloatTensor(DFKs.GRID_Z[0][i]).cuda()
            GRID.pad[i] = torch.FloatTensor(DFKs.GRID_pad[0][i]).cuda()

        PrincDirects = myload_PrincDirects(self.para_folder + '/PrincDirects.mat')

        ## Step 1.2: which is VT3D_rotatDFKStickAlongPrincDirects in MATLAB (line 857)
        GRID_use = Const_strcData()
        GRID_use.X = GRID.X[sel_k_no]
        GRID_use.Y = GRID.Y[sel_k_no]
        GRID_use.Z = GRID.Z[sel_k_no]
        GRID_use.pad = GRID.pad[sel_k_no]

        DFKOrientedSticks = Const_strcData()
        DFKOrientedSticks.Dk = [None] * len(PrincDirects)
        for ors in range(len(PrincDirects)):
            ## Rotating the kernel Data according to the MainOrients
            Dkrot = VTF3D_rotate3DKernel(stickDFK.Dk[sel_k_no], stickDFK.phi[sel_k_no],
                                         PrincDirects[ors])
            ## Cropping out the Padding
            ## Rotated Kerbel Tubularity Prob 'Dkrot'
            Dkrot = Dkrot[int(GRID_use.pad[0]):int(Dkrot.shape[0] - GRID_use.pad[0]),
                    int(GRID_use.pad[1]):int(Dkrot.shape[1] - GRID_use.pad[1]), int(GRID_use.pad[2]):int(Dkrot.shape[2] - GRID_use.pad[2])]
            ## Balancing the kernel response
            Dkrot = Dkrot - (np.sum(Dkrot) / Dkrot.size)  # sum up to zero
            ## Assign and storing Data
            DFKOrientedSticks.Dk[ors] = Dkrot
            del Dkrot

        # Step 2: Convolution
        # Step 2.1: Rename DFKOrientedSticks.Dk into kernel
        # down_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        down_factors = [0.25, 0.5, 0.75, 1.0]
        kernell = [None] * len(DFKOrientedSticks.Dk)
        for i in range(len(DFKOrientedSticks.Dk)):
            kernell[i] = torch.FloatTensor(DFKOrientedSticks.Dk[i])[None, None, ...].cuda()

        ## Unet down-sampling
        all_kernel_image = torch.zeros([1] + (list(self.image_data.shape[-3:])))
        for i in range(1):
            ### Down-sample loop
            all_upDx = torch.zeros([len(down_factors)] + (list(self.image_data.shape[-3:])))
            for j in range(len(down_factors)):
                pool_layer = nn.AdaptiveAvgPool3d((int(down_factors[j] * self.image_data.shape[-3]),
                                                   int(down_factors[j] * self.image_data.shape[-2]),
                                                   int(down_factors[j] * self.image_data.shape[-1]))).cuda()

                pooled_image_data = pool_layer(self.image_data)
                downDx_tmp = F.conv3d(pooled_image_data,
                                      kernell[i].float(),
                                      stride=(1, 1, 1),
                                      padding=(2, 2, 2),
                                      dilation=(1, 1, 1))

                all_upDx[j, ...] = upsamp(downDx_tmp, list(self.image_data.shape[-3:]))
                del pool_layer, downDx_tmp
            all_kernel_image[i, ...], _ = torch.max(all_upDx, dim=0)
            del all_upDx

        slog, _ = torch.max(all_kernel_image, dim=0)

        return slog


# import time
# fake_A_test, aff = read_file(
#                              "/storage/CycleGAN-related/SABRE-MRA-Affine/Images/sub-0_ses-v3_T1.nii.gz")
# # fake_A_test, aff = read_file("/storage/Outputs-MRA-GAN/C1-Figures/"
# #                              "mra-gen2ish-sabre-not1-nopercep-nomsssim-dice-nonoise-5d-0-100-now-25fl-v1/"
# #                              "Fake_A_Vasc_MNI_syn_1477.nii.gz")
# fake_A_test = (fake_A_test - fake_A_test.mean()) / fake_A_test.std()
#
# fake_A_test = fake_A_test[:80, 40:120, :80]
#
#
# fake_A_test = torch.FloatTensor(fake_A_test[None, None, ...]).cuda()
#
# start_time = time.time()
#
# preproc = preprocessing("/home/pedro/MRA-GAN/MRA-GAN/VSeg/MAT_files",
#                         fake_A_test)
#
# slog = preproc.process()
# print(time.time() - start_time)
# # # Save
# random_num = np.random.randint(1000)
# save_img(slog.squeeze().cpu().detach().numpy(), aff, "/storage/Outputs-MRA-GAN/C1-Figures/"
#                                            "mra-gen2ish-sabre-not1-nopercep-nomsssim-dice-nonoise-5d-0-100-now-25fl-v1/"
#                                            f"slog_test_{random_num}.nii.gz")
# save_img(fake_A_test.squeeze().cpu().detach().numpy(), aff, "/storage/Outputs-MRA-GAN/C1-Figures/"
#                                                   "mra-gen2ish-sabre-not1-nopercep-nomsssim-dice-nonoise-5d-0-100-now-25fl-v1/"
#                                                   f"fake_A_test_{random_num}.nii.gz")
# # from monai.networks.nets import UNet
# from chinai.networks.nets import UNet
# from chinai.networks.layers import Norm
#
# vseg_model = UNet(dimensions=3, in_channels=2, out_channels=2,
#                   channels=(16, 32, 64, 128, 256), strides=(1, 1, 1, 1),
#                   num_res_units=2, norm=Norm.BATCH).cuda()
# vseg_model.load_state_dict(torch.load(os.path.join(f"/home/pedro/MRA-GAN/MRA-GAN/VSeg/PretrainedModels",
#                                                    "last_model_Nep2000.pth")), strict=True)
#
# vseg_model.eval()
#
# # # Output segmentation
# seg_fake_A = torch.softmax(vseg_model(torch.cat((fake_A_test, slog[None, None, ...].cuda()),
#                                                 dim=1)), dim=1)
#
# # # Loss
# # # loss_seg_fake_A_loss = criterionDice(seg_fake_A[:, 1, ...][:, None, ...], real_B)
# #
# # # Save, sometimes
# # # if some_iter % 200 == 0:
# save_img(seg_fake_A[:, 1, ...].squeeze().cpu().detach().numpy(),
#          aff,
#          "/storage/Outputs-MRA-GAN/C1-Figures/mra-gen2ish-sabre-not1-nopercep-nomsssim-dice-nonoise-5d-0-100-now-25fl-v1/"
#          f"seg_test_{random_num}.nii.gz",
#          overwrite=True)
