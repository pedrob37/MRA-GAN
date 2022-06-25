from scripts.preprocessing import preprocessing
from scripts.training import training
from scripts.inferring import inferring
from fire import Fire

def run(pwd: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/week02/20210706_Vseg_v2",
        mode: int = 2, # 0 = preprocessing, 1 = training, 2 = infering,
        pp_choice: str = "Max",
        pp_folder: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/Data/CTCTA_test3",
        pp_save_folder: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/Data/CTCTA_test3_SLOGv2",
        pp_tmp_catch: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/catch/catch_20210706_SLOGs",
        pp_filetype: str = "*.nii.gz",
        pp_para_folder: str = "/xxx/parameters",

        tt_image: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/Data/MRA_test3",
        tt_label: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/Data/MRA_test3_label",
        tt_image_k: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/Data/MRA_test3_k",
        tt_image_filetype: str = "*.nii.gz",
        tt_label_filetype: str = "*.nii.gz",
        tt_image_k_filetype: str = "*.nii.gz",
        tt_numval: int = 1,
        tt_tmp_catch_train: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/catch/catch_20210706_Unet_train",
        tt_tmp_catch_val: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/catch/catch_20210706_Unet_test",
        tt_Nepoch: int = 1000,
        tt_patch_size: int = 20,
        tt_ROI_val: int = 40,
        tt_num_samples: int = 4,
        tt_mainsavefold: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/week02/20210706_Vseg_v2",

        ii_image: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/Data/CTCTA_test3",
        ii_image_k: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/Data/CTCTA_test3_SLOG",
        ii_image_filetype: str = "*.nii.gz",
        ii_image_k_filetype: str = "*.nii.gz",
        ii_load_model_folder: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp/NewEpoch_2000",
        ii_model_file: str = "last_model_Nep2000.pth",
        ii_save_folder: str = 'infer_out',
        ii_ROI_val: int = 80,
        ii_tmp_catch: str = "/home/chayanin/PycharmProjects/2020_v08_vrienv007/catch/catch_20210706_Unet_infer",
       ):
    if mode == 0:
        print('pwd', pwd)
        print('mode', mode)
        print('pp_choice', pp_choice)
        print('pp_folder', pp_folder)
        print('pp_save_folder', pp_save_folder)
        print('pp_tmp_catch', pp_tmp_catch)
        print('pp_filetype', pp_filetype)
        print('pp_para_folder', pp_para_folder)

        PP = preprocessing(pwd, pp_choice, pp_folder, pp_save_folder, pp_tmp_catch, pp_filetype, pp_para_folder)
        PP.process()
        del PP
    if mode == 1:
        print('pwd', pwd)
        print('mode', mode)
        print('tt_image', tt_image)
        print('tt_label', tt_label)
        print('tt_image_k', tt_image_k)
        print('tt_image_filetype', tt_image_filetype)
        print('tt_label_filetype', tt_label_filetype)
        print('tt_image_k_filetype', tt_image_k_filetype)
        print('tt_tmp_catch_train', tt_tmp_catch_train)
        print('tt_tmp_catch_val', tt_tmp_catch_val)
        print('tt_numval', tt_numval)
        print('tt_Nepoch', tt_Nepoch)
        print('tt_patch_size', tt_patch_size)
        print('tt_ROI_val', tt_ROI_val)
        print('tt_num_samples', tt_num_samples)
        print('tt_mainsavefold', tt_mainsavefold)

        TT = training(pwd, tt_image, tt_label, tt_image_k, tt_image_filetype, tt_label_filetype, tt_image_k_filetype, tt_tmp_catch_train, tt_tmp_catch_val, tt_numval, tt_Nepoch, tt_patch_size, tt_ROI_val, tt_num_samples, tt_mainsavefold)
        TT.train()
        del TT
    if mode == 2:
        print('pwd', pwd)
        print('mode', mode)
        print('ii_image', ii_image)
        print('ii_image_k', ii_image_k)
        print('ii_image_filetype', ii_image_filetype)
        print('ii_image_k_filetype', ii_image_k_filetype)
        print('ii_load_model_folder', ii_load_model_folder)
        print('ii_model_file', ii_model_file)
        print('ii_save_folder', ii_save_folder)
        print('ii_ROI_val', ii_ROI_val)
        print('ii_tmp_catch', ii_tmp_catch)

        II = inferring(pwd, ii_image, ii_image_k, ii_image_filetype, ii_image_k_filetype, ii_load_model_folder, ii_model_file, ii_save_folder, ii_ROI_val, ii_tmp_catch)
        II.infer()
        del II
    x = 0
    return x

if __name__ == "__main__":
    Fire(run)