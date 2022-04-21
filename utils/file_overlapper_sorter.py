import os
import shutil


# Moving files function
def file_mover(file_name, original_directory, final_directory):
    file_basename = os.path.basename(file_name)
    shutil.move(os.path.join(original_directory,
                             file_basename),
                os.path.join(final_directory,
                             file_basename))


# Original, unorganized directories
images_directory = "/storage/CycleGAN-related/IXI/Affine-Triple-IXI-MRA-UnNorm/All_Images"
labels_directory = "/storage/CycleGAN-related/IXI/Affine-Triple-IXI-MRA-UnNorm/All_Labels"
t1_directory = "/storage/CycleGAN-related/IXI/Affine-Triple-IXI-MRA-UnNorm/All_T1s"

# Organized directories
final_images_directory = "/storage/CycleGAN-related/IXI/Affine-Triple-IXI-MRA-UnNorm/Images"
final_labels_directory = "/storage/CycleGAN-related/IXI/Affine-Triple-IXI-MRA-UnNorm/Labels"
final_t1_directory = "/storage/CycleGAN-related/IXI/Affine-Triple-IXI-MRA-UnNorm/T1s"

# Find list of files
images_set = set(os.listdir(images_directory))
labels_set = set(os.listdir(labels_directory))
t1_set = set(os.listdir(t1_directory))

# Find intersects
intersects = images_set.intersection(labels_set, t1_set)

# Moving relevant files to relevant directories
for file in intersects:
    file_mover(file, images_directory, final_images_directory)
    file_mover(file, labels_directory, final_labels_directory)
    file_mover(file, t1_directory, final_t1_directory)
