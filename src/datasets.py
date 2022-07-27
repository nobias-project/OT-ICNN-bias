#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:14:17 2022

@author: simonefabrizzi
"""

import os
import torch
import pandas as pd

from torch.utils import data
from skimage import io


class CelebA(data.Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.celeba = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.celeba.loc[idx, "image_id"])
        image = io.imread(img_path)
        

        if self.transform:
            image = self.transform(image)

        return image,  idx, self.celeba.loc[idx, "image_id"]