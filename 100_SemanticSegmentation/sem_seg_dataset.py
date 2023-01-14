from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2

class SegmentationDataset(Dataset):
    """Create Semantic Segmentation Dataset. Read images, apply augmentations, and process transformations

    Args:
        Dataset (image): Aerial Drone Images
    """
    # CLASSES = {'building': 44, 'land': 91, 'road':172, 'vegetation':212, 'water':171, 'unlabeled':155}
    # CLASSES_KEYS = list(CLASSES.keys())
    
    def __init__(self, path_name) -> None:
        super().__init__()
        self.image_names = os.listdir(f"{path_name}/images")
        self.image_paths = [f"{path_name}/images/{i}" for i in self.image_names]
        self.masks_names = os.listdir(f"{path_name}/masks")
        self.masks_paths = [f"{path_name}/masks/{i}" for i in self.masks_names]
        
        # filter all images that do not exist in both folders
        self.img_stem = [Path(i).stem for i in self.image_paths]
        self.msk_stem = [Path(i).stem for i in self.masks_paths]
        self.img_msk_stem = set(self.img_stem) & set(self.msk_stem)
        self.image_paths = [i for i in self.image_paths if (Path(i).stem in self.img_msk_stem)]


    def convert_mask(self, mask):
        mask[mask == 155] = 0  # unlabeled
        mask[mask == 44] = 1  # building
        mask[mask == 91] = 2  # land
        mask[mask == 171] = 3  # water
        mask[mask == 172] = 4  # road
        mask[mask == 212] = 5  # vegetation
        return mask   

    def __len__(self):
        return len(self.img_msk_stem)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))  #structure: BS, C, H, W
        mask =  cv2.imread(self.masks_paths[index], 0)
        mask = self.convert_mask(mask)
        return image, mask
    