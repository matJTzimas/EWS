#!/usr/bin/env python3
import os
import shutil
import argparse
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import yaml
from types import SimpleNamespace


# ---------------------------------------------------------------------
# Utility Function: Copy files from one folder to another.
# ---------------------------------------------------------------------
def copy_files(source_folder, destination_folder):
    """
    Copies all files from source_folder to destination_folder.
    """
    os.makedirs(destination_folder, exist_ok=True)
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        if os.path.isfile(source_file):
            destination_file = os.path.join(destination_folder, filename)
            shutil.copy2(source_file, destination_file)
            # Uncomment the next line for verbose output.
            # print(f"Copied: {source_file} to {destination_file}")

# ---------------------------------------------------------------------
# Define a Label class to structure the label information.
# ---------------------------------------------------------------------
class Label:
    def __init__(self, name, id, trainId, category, catId, hasInstances, ignoreInEval, color):
        self.name = name
        self.id = id
        self.trainId = trainId
        self.category = category
        self.catId = catId
        self.hasInstances = hasInstances
        self.ignoreInEval = ignoreInEval
        self.color = color

# ---------------------------------------------------------------------
# List of labels (modify as needed).
# ---------------------------------------------------------------------
labels = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# ---------------------------------------------------------------------
# Function to process a single segmentation file.
# ---------------------------------------------------------------------
def process_single_file(task):
    """
    task: a tuple (in_path, out_path, id_to_catId, category_ids)
    """
    in_path, out_path, id_to_catId, category_ids = task
    try:
        # Load the segmentation mask.
        mask = Image.open(in_path)
        mask_array = np.array(mask)
        # Convert each pixel value using the mapping from label id to category id.
        new_mask = np.vectorize(id_to_catId.get)(mask_array)
        # Create a binary mask: pixels that match any target category become 1.
        binary_mask = np.zeros_like(new_mask, dtype=np.uint8)
        for cat_id in category_ids:
            binary_mask[new_mask == cat_id] = 1
        # Save the binary mask.
        binary_mask_image = Image.fromarray(binary_mask)
        binary_mask_image.save(out_path)
    except Exception as e:
        print(f"Error processing file {in_path}: {e}")

# ---------------------------------------------------------------------
# Process segmentation masks concurrently for a given folder.
# ---------------------------------------------------------------------
def process_segmentation_masks_cls_concurrent(input_base_folder, output_folder, category_name):
    """
    Processes segmentation masks found in the subdirectories of input_base_folder,
    converting them to binary masks for the given category_name and saving them in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a mapping from label id to category id.
    id_to_catId = {label.id: label.catId for label in labels}
    # Determine which category ids correspond to the target category.
    category_ids = {label.catId for label in labels if label.category == category_name}
    
    tasks = []
    # Loop over each city folder.
    for city in sorted(os.listdir(input_base_folder)):
        city_path = os.path.join(input_base_folder, city)
        if os.path.isdir(city_path):
            for filename in os.listdir(city_path):
                if filename.endswith('labelIds.png'):  # Only process files that end with 'labelIds.png'
                    in_file = os.path.join(city_path, filename)
                    out_file = os.path.join(output_folder, filename)
                    tasks.append((in_file, out_file, id_to_catId, category_ids))
    
    print(f"Found {len(tasks)} tasks in {input_base_folder} to process.")

    # Process files concurrently.
    with ProcessPoolExecutor() as executor:
        list(executor.map(process_single_file, tasks))

# ---------------------------------------------------------------------
# Main function to create the one-class Cityscapes dataset ("CityStego").
# ---------------------------------------------------------------------
def create_one_cityscapes(cityscapes_base, output_base, category_name='vehicle'):
    """
    cityscapes_base: Base folder of the Cityscapes dataset.
    output_base: Folder where the new dataset (CityStego) will be stored.
    category_name: The target category to extract (default is 'vehicle').
    """
    # Define input paths for images and labels.
    leftImg8bit_train = os.path.join(cityscapes_base, 'leftImg8bit', 'train')
    leftImg8bit_val   = os.path.join(cityscapes_base, 'leftImg8bit', 'val')
    gtFine_train      = os.path.join(cityscapes_base, 'gtFine', 'train')
    gtFine_val        = os.path.join(cityscapes_base, 'gtFine', 'val')

    output_base = os.path.join(output_base,'CityscapesOne')
    os.makedirs(output_base, exist_ok=True)
    
    # Define output directories.
    imgs_train   = os.path.join(output_base, 'imgs', 'train')
    imgs_val     = os.path.join(output_base, 'imgs', 'val')
    labels_train = os.path.join(output_base, 'labels', 'train')
    labels_val   = os.path.join(output_base, 'labels', 'val')

    # Create output subdirectories if they do not exist.
    os.makedirs(imgs_train, exist_ok=True)
    os.makedirs(imgs_val, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Copy Train Images.
    # -----------------------------------------------------------------
    for city in sorted(os.listdir(leftImg8bit_train)):
        curr_path = os.path.join(leftImg8bit_train, city)
        if os.path.isdir(curr_path):
            copy_files(curr_path, imgs_train)
            print(f'Train images copied from {curr_path}')
    print('ALL Train images copied.')

    # -----------------------------------------------------------------
    # 2. Copy Validation Images.
    # -----------------------------------------------------------------
    for city in sorted(os.listdir(leftImg8bit_val)):
        curr_path = os.path.join(leftImg8bit_val, city)
        if os.path.isdir(curr_path):
            copy_files(curr_path, imgs_val)
            print(f'Val images copied from {curr_path}')
    print('ALL Val images copied.')

    # -----------------------------------------------------------------
    # 3. Process Train Labels.
    # -----------------------------------------------------------------
    process_segmentation_masks_cls_concurrent(gtFine_train, labels_train, category_name)
    print("All Train labels processed and transformed.")

    # -----------------------------------------------------------------
    # 4. Process Validation Labels.
    # -----------------------------------------------------------------
    process_segmentation_masks_cls_concurrent(gtFine_val, labels_val, category_name)
    print("All Val labels processed and transformed.")

# ---------------------------------------------------------------------
# Entry point: Parse command-line arguments and run the processing.
# ---------------------------------------------------------------------
if __name__ == '__main__':

    file_path = './configs/train_config.yaml'

    # Reading the YAML file
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    cfg = SimpleNamespace(**config_dict)


    create_one_cityscapes(
        cityscapes_base=cfg.raw_dataset_path,
        output_base=cfg.pytorch_data_dir,
        category_name=cfg.city_category
    )