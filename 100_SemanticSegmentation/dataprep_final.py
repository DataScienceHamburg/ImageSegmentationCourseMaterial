# %% Purpose: 
# 1. Prepare images and masks folder within train, val, test folder
# 2. copy all images into these folders
# 3. create patches of images
#%% Packages
import os
import re
from pathlib import Path
import numpy as np
from patchify import patchify
from PIL import Image
#%% Create empty folders if necessary
def create_folders():
    FOLDERS = ['train', 'val', 'test']
    for folder in FOLDERS:
        if not os.path.exists(folder):
            folder_imgs = f"{folder}/images"
            folder_msks = f"{folder}/masks"
            os.makedirs(folder_imgs) if not os.path.exists(folder_imgs) else print('folder already exists')
            os.makedirs(folder_msks) if not os.path.exists(folder_msks) else print('folder already exists')

create_folders()

#%% create patches
# %% PATCHES
def create_patches(src, dest_path):
    path_split = os.path.split(src)
    tile_num = re.findall(r'\d+', path_split[0])[0]
    
    image = Image.open(src)
    image = np.asarray(image)
    if len(image.shape) > 2:  # only if color channel exists as well
        patches = patchify(image, (320, 320, 3), step=300)
        file_name_wo_ext = Path(src).stem
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                patch = Image.fromarray(patch)
                num = i * patches.shape[1] + j
                patch.save(f"{dest_path}/{file_name_wo_ext}_tile_{tile_num}_patch_{num}.png")

#%% copy all files
for path_name, _, file_name in os.walk('data'):
    for f in file_name:
        print(f)
        if f != 'classes.json':
            
            path_split = os.path.split(path_name)
            tile_num = re.findall(r'\d+', path_split[0])[0]
            
            img_type =path_split[1]  # either 'masks' or 'images'
            
            # leave out tile 2, issues with color dim
            if tile_num == '3':
                target_folder_imgs = 'val'
                target_folder_masks = 'val'
            elif tile_num == '1':
                target_folder_imgs = 'test'
                target_folder_masks = 'test'
            elif tile_num in ['4', '5', '6', '7', '8']:
                target_folder_imgs = 'train'
                target_folder_masks = 'train'
            
            # copy all images
            src = os.path.join(path_name, f)
            file_name_wo_ext = Path(src).stem
            # check if file exists in images and masks
            img_file = f"{path_split[0]}/images/{file_name_wo_ext}.jpg"
            mask_file = f"{path_split[0]}/masks/{file_name_wo_ext}.png"
            if os.path.exists(img_file) and os.path.exists(mask_file):
                if img_type == 'images':
                    dest = os.path.join(target_folder_imgs, img_type)
                    create_patches(src=src, dest_path=dest)
                    
                    
                
                # copy all masks
                if img_type == 'masks':
                    dest = os.path.join(target_folder_masks, img_type)
                    create_patches(src=src, dest_path=dest)
                             
# %%
