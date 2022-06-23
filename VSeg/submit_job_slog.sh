sudo docker run --runtime=nvidia --rm -v /home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/data_tmp:/new_data \
 -i 65e1fc2d20d1 \
 --pwd="/home/chayanin/PycharmProjects/2020_v08_vrienv007/2021/2021_07_July/week02/20210709_vseg" \
 --mode=0 \
 --pp_choice="Max" \
 --pp_folder="/new_data/Data/CTCTA_test3" \
 --pp_save_folder="/new_data/Data/CTCTA_test3_SLOGv2" \
 --pp_tmp_catch="/new_data/catch_20210706_SLOGs" \
 --pp_filetype="*.nii.gz" \
 --pp_para_folder="/new_data/Data/parameters"
