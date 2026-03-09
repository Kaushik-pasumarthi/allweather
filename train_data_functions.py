import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
from torchvision import transforms
import os


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, train_filename):
        super().__init__()
        train_list = train_data_dir + train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]
            # NEW: Create mask paths
            mask_names = [i.strip().replace('input', 'masks') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.mask_names = mask_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        mask_name = self.mask_names[index]

        img_id = re.split('/', input_name)[-1][:-4]

        input_img = Image.open(self.train_data_dir + input_name).convert('RGB')
        gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        # NEW: Load the mask as grayscale ('L')
        mask_img = Image.open(self.train_data_dir + mask_name).convert('L')

        width, height = input_img.size

        # Resize logic applied to all three
        if width < crop_width and height < crop_height:
            input_img = input_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            mask_img = mask_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width:
            input_img = input_img.resize((crop_width, height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, height), Image.ANTIALIAS)
            mask_img = mask_img.resize((crop_width, height), Image.ANTIALIAS)
        elif height < crop_height:
            input_img = input_img.resize((width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)
            mask_img = mask_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        # Generate exact same crop coordinates for all
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)

        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        mask_crop_img = mask_img.crop((x, y, x + crop_width, y + crop_height))  # NEW

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt_mask = Compose([ToTensor()])  # Handles both GT and Mask

        input_im = transform_input(input_crop_img)
        gt = transform_gt_mask(gt_crop_img)
        mask = transform_gt_mask(mask_crop_img)  # NEW: tensor of shape [1, H, W]

        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        # NEW: Return the mask alongside everything else
        return input_im, gt, mask, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


# --- All Weather Dataset (The cleaner one) --- #
class AllWeatherDataset(data.Dataset):
    def __init__(self, root, file_list, crop_size, train=True):
        self.root = root
        self.crop_size = crop_size
        self.train = train

        with open(file_list, 'r') as f:
            self.input_files = [l.strip() for l in f.readlines()]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        inp_rel = self.input_files[idx].replace('./allweather/', '')

        inp_path = os.path.join(self.root, inp_rel)
        gt_path = inp_path.replace('input', 'gt')
        mask_path = inp_path.replace('input', 'masks')  # NEW: Target the masks folder

        inp = Image.open(inp_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # NEW: Load mask in Grayscale

        # -------- EXACT MATCH CROPPING -------- #
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(inp, self.crop_size)
        else:
            w_img, h_img = inp.size
            th, tw = self.crop_size
            i = (h_img - th) // 2
            j = (w_img - tw) // 2
            h, w = th, tw

        inp = TF.crop(inp, i, j, h, w)
        gt = TF.crop(gt, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)  # NEW: Crop mask exactly the same way
        # -------------------------------------- #

        inp = TF.to_tensor(inp)
        gt = TF.to_tensor(gt)
        mask = TF.to_tensor(mask)  # NEW

        # NEW: Return mask
        return inp, gt, mask, inp_rel