from utils.utils import *
from VSeg.scripts.reworked_preprocessing import preprocessing
from chinai.networks.nets import UNet
from chinai.networks.layers import Norm
from chinai.data import sliding_window_inference
import argparse

parser = argparse.ArgumentParser(description='Passing files + relevant directories')
parser.add_argument('--base_dir', type=str, default="/nfs/home/pedro")
parser.add_argument('--images_dir', type=str, default="/nfs/home/pedro")
arguments = parser.parse_args()

# Base directory
base_dir = arguments.base_dir

# Process Princ + DFK files first: Only needs to be done once
preproc = preprocessing(f"{base_dir}/MRA-GAN/VSeg/MAT_files")
dfk_file = preproc.process_DFK()

# List files to be segmented: IXI
images_dir = arguments.images_dir
os.chdir(images_dir)
file_list = os.listdir(images_dir)

# Filter, just in case
file_list = [x for x in file_list if "nii.gz" in x]

# VSeg network
vseg_model = UNet(dimensions=3, in_channels=2, out_channels=2,
                  channels=(16, 32, 64, 128, 256), strides=(1, 1, 1, 1),
                  num_res_units=2, norm=Norm.BATCH).cuda()
vseg_model.load_state_dict(torch.load(os.path.join(f"{base_dir}/MRA-GAN/VSeg/PretrainedModels",
                                                   "last_model_Nep2000.pth")), strict=True)
vseg_model.eval()

# Process files, loop
for mra in file_list:
    vol, aff = read_file(mra)
    vol = (vol - vol.mean()) / vol.std()
    vol = torch.FloatTensor(vol[None, None, ...]).cuda()
    slog = preproc.process_conv(vol, dfk_file)

    # # Output segmentation
    inf_outputs = sliding_window_inference(torch.cat((vol, slog[None, None, ...].cuda()),
                                                     dim=1), vol.shape, 1, vseg_model)
    seg_fake_A = torch.softmax(inf_outputs, dim=1)

    # Save
    save_img(seg_fake_A[:, 1, ...].squeeze().cpu().detach().numpy(),
             aff,
             os.path.join(arguments.images_dir, 'Segs', f"Seg_{os.path.basename(mra)}"),
             overwrite=True)

    break
