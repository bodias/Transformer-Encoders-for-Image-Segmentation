import os
import torch
import shutil
import numpy as np
import pandas as pd

from PIL import Image
from imgutils import separate_blobs


class BrightfieldDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", masks_directory='complete_stencils', split_file="", transform=None):

        assert mode in {"train", "val", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, masks_directory)

        self.filenames = self._read_split(split_file)  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        image_filename, mask_filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, image_filename[image_filename.rfind("/")+1:])
        mask_path = os.path.join(self.masks_directory, mask_filename[mask_filename.rfind("/")+1:])

        image = Image.open(image_path)#.resize((256, 256), Image.LINEAR)
        mask = Image.open(mask_path)#.resize((256, 256), Image.LINEAR)

        image = np.array(image, dtype=np.uint16)
        mask = np.array(mask, dtype=np.uint16)
        mask = self._preprocess_mask(mask)        
        
        sample = dict(image=image, mask=mask)
        if self.transform is not None:
            sample = self.transform(**sample)

        # convert to other format HWC -> CHW
        image = np.moveaxis(sample['image'], -1, 0)
        image = np.expand_dims(image, 0)    
        mask = np.expand_dims(sample['mask'], 0)       

        return dict(image=image.astype(np.float32), mask=mask.astype(np.float32))

    @staticmethod
    def _preprocess_mask(mask):     
        # mask = mask.astype(np.float32)
        # make all cell classes being represented by the same class (1) and background is 0
        # separate_blobs(mask)
        mask[mask>0] = 1
        
        return mask

    def _read_split(self, split_file):
        split_data = pd.read_csv(split_file, index_col=0)        
        filenames = list(zip(split_data[split_data.set==self.mode].bf.values, split_data[split_data.set==self.mode].stencil.values))

        return filenames

    