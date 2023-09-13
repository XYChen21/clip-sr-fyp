import os
import random
import shutil

# Path to the folder containing your images
image_folder = '/data/xychen/DF2K/DF2K_train_HR_sub'

# Path to the folder where you want to save the selected images (optional)
output_folder = '/data/xychen/DF2K/test'

# Number of images to select
num_images_to_select = 100

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Randomly select 100 images
selected_images = random.sample(image_files, num_images_to_select)

# Copy the selected images to the output folder (optional)
if output_folder:
    os.makedirs(output_folder, exist_ok=True)
    for image in selected_images:
        shutil.copy(os.path.join(image_folder, image), os.path.join(output_folder, image))
