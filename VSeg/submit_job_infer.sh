sudo docker run --runtime=nvidia --rm -v /home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp:/new_data \
 -i 90bcad71a8e5 \
 --pwd="/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/week02v2/20210709_vseg" \
 --mode=2 \
 --ii_image="/new_data/Data/CTCTA_test3" \
 --ii_image_k="/new_data/Data/CTCTA_test3_SLOG" \
 --ii_image_filetype="*.nii.gz" \
 --ii_image_k_filetype="*.nii.gz" \
 --ii_load_model_folder="/new_data/NewEpoch_2000" \
 --ii_model_file="last_model_Nep2000.pth" \
 --ii_save_folder='infer_out2' \
 --ii_ROI_val=80 \
 --ii_tmp_catch="/new_data/catch_inferUnet_20210713"