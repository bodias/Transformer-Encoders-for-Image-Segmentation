import os
import torch
import shutil
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve


class BrightfieldDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", masks_directory='complete_stencils', mask_threshold=0.01, split_file="", transform=None):

        assert mode in {"train", "val", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform
        self.mask_threshold = mask_threshold

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, masks_directory)

        self.filenames = self._read_split(split_file)  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        image_filename, mask_filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, image_filename[image_filename.rfind("/")+1:])
        mask_path = os.path.join(self.masks_directory, mask_filename[mask_filename.rfind("/")+1:])
        # print(mask_path)

        image = Image.open(image_path).resize((256, 256), Image.LINEAR)
        mask = Image.open(mask_path).resize((256, 256), Image.LINEAR)

        image = np.array(image)
        mask = np.array(mask)
        mask = self._preprocess_mask(mask, self.mask_threshold)
        
        # convert to other format HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        image = np.expand_dims(image, 0)    
        mask = np.expand_dims(mask, 0)        
        
        # if self.transform is not None:
        #     image = self.transform(image)
        #     mask  = self.transform(mask)

        sample = dict(image=image, mask=mask)    

        return sample

    @staticmethod
    def _preprocess_mask(mask, threshold):     
        mask = mask.astype(np.float32)           
        norm_mask = mask - np.min(mask) 
        norm_mask = norm_mask / (np.max(mask) - np.min(mask))

        mask[mask<=threshold] = 0.0
        mask[mask>threshold] = 1.0
        
        return mask

    def _read_split(self, split_file):
        split_data = pd.read_csv(split_file, index_col=0)        
        filenames = list(zip(split_data[split_data.set==self.mode].bf.values, split_data[split_data.set==self.mode].stencil.values))

        return filenames

    # @staticmethod
    # def download(root):

    #     # load images
    #     filepath = os.path.join(root, "images.tar.gz")
    #     download_url(
    #         url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    #         filepath=filepath,
    #     )
    #     extract_archive(filepath)

    #     # load annotations
    #     filepath = os.path.join(root, "annotations.tar.gz")
    #     download_url(
    #         url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
    #         filepath=filepath,
    #     )
    #     extract_archive(filepath)


# class SimpleOxfordPetDataset(OxfordPetDataset):
#     def __getitem__(self, *args, **kwargs):

#         sample = super().__getitem__(*args, **kwargs)

#         # resize images
#         image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
#         mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
#         trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

#         # convert to other format HWC -> CHW
#         sample["image"] = np.moveaxis(image, -1, 0)
#         sample["mask"] = np.expand_dims(mask, 0)
#         sample["trimap"] = np.expand_dims(trimap, 0)

#         return sample


# class TqdmUpTo(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, filepath):
#     directory = os.path.dirname(os.path.abspath(filepath))
#     os.makedirs(directory, exist_ok=True)
#     if os.path.exists(filepath):
#         return

#     with TqdmUpTo(
#         unit="B",
#         unit_scale=True,
#         unit_divisor=1024,
#         miniters=1,
#         desc=os.path.basename(filepath),
#     ) as t:
#         urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
#         t.total = t.n


# def extract_archive(filepath):
#     extract_dir = os.path.dirname(os.path.abspath(filepath))
#     dst_dir = os.path.splitext(filepath)[0]
#     if not os.path.exists(dst_dir):
#         shutil.unpack_archive(filepath, extract_dir)
