import torch
import torch.nn.functional as F
import numpy as np
import os
import nibabel as nib


def upsamp(data5D, s_size):
    x = torch.squeeze(data5D[0, 0, :, :, :])  # torch.randn(3, 4, 5)
    # interpolation on D3 (aka D5)
    x = F.interpolate(x, size=s_size[2])

    # interpolation on D2 (aka D4)
    x = x.permute(0, 2, 1)
    x = F.interpolate(x, size=s_size[1])
    x = x.permute(0, 2, 1)

    # interpolation on D1 (aka D3)
    x = x.permute(2, 1, 0)
    x = F.interpolate(x, size=s_size[0])
    x = x.permute(2, 1, 0)

    data5D_new = torch.randn((1, 1, x.shape[0], x.shape[1], x.shape[2]))
    data5D_new[0, 0, :, :, :] = x

    return data5D_new


def savesum(sum_data, NIIname, savefold, affine_co):
    sum_data = np.float32(sum_data)
    sum_data = sum_data - sum_data.min()  # normalisation
    sum_data = sum_data / sum_data.max()  # normalisation
    save_img = nib.Nifti1Image(sum_data, affine=affine_co)
    print(savefold + '/' + NIIname)
    if os.path.exists(savefold):
        print("path exists.")
        nib.save(save_img, savefold + '/' + NIIname)
    else:
        os.mkdir(savefold)
        nib.save(save_img, savefold + '/' + NIIname)
    return
