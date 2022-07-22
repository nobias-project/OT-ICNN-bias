#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:55:17 2022

@author: simonefabrizzi
"""

from __future__ import print_function
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import src.optimal_transport_modules
from src.optimal_transport_modules.icnn_modules import *
import time
import random
import facenet_pytorch as facenet
import numpy as np
import pandas as pd
import os
import logging
import skimage
import torch.utils.data
import src.datasets
from src.utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import decomposition
from scipy.stats import truncnorm
# from torchsummary import summary
# from torchsummary import summary

parser = argparse.ArgumentParser(description='PyTorch CelebA Toy Beard Experiment Evaluation')
parser.add_argument('--epoch', type=int, default=16, metavar='S',
                    help='epoch to be evaluated')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

results_save_path = "../results/Results_CelebA_Toy_Beard/input_dim_512/init_trunc_inv_sqrt/layers_3/neuron_1024/lambda_cvx_0.1_mean_0.0/optim_Adamlr_0.001betas_0.5_0.99/gen_16/batch_60/trial_1_last_inp_qudr"
model_save_path = results_save_path + '/storing_models'



df = pd.read_csv("../data/celeba/celebA_sample_male.csv")
df["values"] = [None]*len(df)
Features_X = facenet.InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(160)])

convex_f = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(512, 1024, "leaky_relu")
convex_f.load_state_dict(torch.load(model_save_path+'/convex_f_epoch_{}.pt'.format(args.epoch)))

for idx in df.index:
    path = "../data/celeba/Img_folder/Img/" + df.loc[idx, "image_id"]
    img = skimage.io.imread(path)
    img = transform(img)
    features = Features_X(img.reshape(1,3,195,160))
    val = convex_f(features).item()
    
    df.loc[idx, "values"] = val

                                      
df = df.sort_values(by = "values", ascending = False)
imgs = df["image_id"][:36]
array_img_vectors = np.array([skimage.io.imread("../data/celeba/Img_folder/Img/" + file) for file in imgs])

def save_images_as_grid(array_img_vectors, epoch):

    #array_img_vectors is of size (N, PCA_components). So obtain the images first using inverse PCA transform

    array_img_vectors = torch.from_numpy(array_img_vectors).float().permute(0,3, 1, 2)
    grid = make_grid(array_img_vectors, nrow=6, normalize=True)*255
    print(grid.shape)
    ndarr = grid.to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr[0])

    im.save(results_save_path+'/grid_epoch_{}.jpeg'.format(epoch))


save_images_as_grid(array_img_vectors, args.epoch)
