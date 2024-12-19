import glob
import os
from PIL import Image
import random
import numpy as np

from torch import nn
import torch
import math
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

from .realesrgan import RealESRGAN_degradation

class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []

        root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder +'/sr_bicubic'
            tag_path = root_folder +'/tag'
            gt_path = root_folder +'/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))
            self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))


        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):

       
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.tag_path_list[index]
            file = open(tag_path, 'r')
            tag = file.read()
            file.close()

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)

        lq_img = lq_img.squeeze()

        ram_values = F.interpolate(lq_img.unsqueeze(0), size=(384, 384), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        return example

    def __len__(self):
        return len(self.gt_list)


class WaymoRepairDataset(data.Dataset):
    def __init__(self,
                 root_folders=None,
                 tokenizer=None,
                 null_text_ratio=0.5):
        super().__init__()
        self.crop_preproc = transforms.Compose([
            transforms.RandomCrop((512, 512)),
            transforms.RandomHorizontalFlip(),
        ])

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.data_dir = root_folders
        self.tokenizer = tokenizer
        self.null_text_ratio = null_text_ratio
        self.gt_dir = os.path.join(self.data_dir, 'gt')
        self.lq_dir = os.path.join(self.data_dir, 'lq')
        self.tag_dir = os.path.join(self.data_dir, 'tag')
        self.gt_list = sorted(glob.glob(os.path.join(self.gt_dir, '*.png')))
        self.lq_list = sorted(glob.glob(os.path.join(self.lq_dir, '*.png')))
        self.tag_list = sorted(glob.glob(os.path.join(self.tag_dir, '*.txt')))

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_path = self.gt_list[idx]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)

        lq_path = self.lq_list[idx]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        assert gt_img.shape == lq_img.shape

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.tag_path_list[idx]
            file = open(tag_path, 'r')
            tag = file.read()
            file.close()

        img = torch.cat([gt_img, lq_img], dim=0)
        if img.shape[1] < 512 or img.shape[2] < 512:
            # 等比缩放
            scale_factor = 512 / min(img.shape[1], img.shape[2])
            new_height = math.ceil(img.shape[1] * scale_factor)
            new_width = math.ceil(img.shape[2] * scale_factor)
            img = F.resize(img, (new_height, new_width),
                                 interpolation=transforms.InterpolationMode.BICUBIC)
        img = self.crop_preproc(img)

        gt_img, lq_img = img[:3, :, :], img[3:, :, :]

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)

        lq_img = lq_img.squeeze()

        ram_values = F.interpolate(lq_img.unsqueeze(0), size=(384, 384), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        return example

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids