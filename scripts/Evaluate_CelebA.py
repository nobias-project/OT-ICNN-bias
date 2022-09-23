#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:55:17 2022

@author: simonefabrizzi
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import random
from src.optimal_transport_modules.icnn_modules import *
import numpy as np
import pandas as pd
import torch.utils.data
import src.datasets
from src.utils import *
from PIL import Image


parser = argparse.ArgumentParser(description='Experiment1 Evaluation')

parser.add_argument('--DATASET_Y',
                    type=str,
                    default=("../data/celeba/"
                             "experiment1_Female_Eyeglasses.csv"),
                    help='X data')

parser.add_argument('--DATASET_X',
                    type=str,
                    default=("../data/celeba/"
                             "experiment1_Male_Eyeglasses_90%.csv"),
                    help='Y data')

parser.add_argument('--FEATURES',
                    type=str,
                    default="resnet18",
                    help='Features extractor')

parser.add_argument('--INPUT_DIM',
                    type=int,
                    default=512,
                    help='dimensionality of the input x')

parser.add_argument('--BATCH_SIZE',
                    type=int,
                    default=30,
                    help='size of the batches')

parser.add_argument('--epoch',
                    type=int,
                    default=30,
                    metavar='S',
                    help='number_of_epochs')

parser.add_argument('--N_GENERATOR_ITERS',
                    type=int,
                    default=5,
                    help='number of training steps for discriminator per iter')

parser.add_argument('--NUM_NEURON',
                    type=int,
                    default=512,
                    help='number of neurons per layer')

parser.add_argument('--NUM_LAYERS',
                    type=int,
                    default=4,
                    help='number of hidden layers before output')

parser.add_argument('--full_quadratic',
                    type=bool,
                    default=False,
                    help='if the last layer is full quadratic or not')

parser.add_argument('--activation',
                    type=str,
                    default='leaky_relu',
                    help='which activation to use for')

parser.add_argument('--initialization',
                    type=str,
                    default='trunc_inv_sqrt',
                    help='which initialization to use for')

parser.add_argument('--TRIAL',
                    type=int,
                    default=1,
                    help='the trail no.')

parser.add_argument('--optimizer',
                    type=str,
                    default='Adam',
                    help='which optimizer to use')

parser.add_argument('--LR',
                    type=float,
                    default=1e-3,
                    help='learning rate')

parser.add_argument('--momentum',
                    type=float,
                    default=0.0,
                    metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.99)


# Less frequently used training settings
parser.add_argument('--LAMBDA_CVX',
                    type=float,
                    default=0.1,
                    help='Regularization constant for '
                    'positive weight constraints')
parser.add_argument('--LAMBDA_MEAN',
                    type=float,
                    default=0.0,
                    help='Regularization constant for '
                    'matching mean and covariance')

parser.add_argument('--log-interval',
                    type=int,
                    default=10,
                    metavar='N',
                    help='how many batches to wait '
                    'before logging training status')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--N_PLOT',
                    type=int,
                    default=16,
                    help='number of samples for plotting')

parser.add_argument('--SCALE',
                    type=float,
                    default=10.0,
                    help='scale for the gaussian_mixtures')
parser.add_argument('--VARIANCE',
                    type=float,
                    default=0.5,
                    help='variance for each mixture')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mps = False  # torch.backends.mps.is_available()

args.lr_schedule = 2  # if args.BATCH_SIZE == 60 else 4

# Seed stuff
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
# understand how to set the seed for mps

np.random.seed(args.seed)
random.seed(args.seed)

# Storing stuff
attribute = args.DATASET_X.split("_")[-2]
percentage = args.DATASET_X[-7:-5]

if args.optimizer == 'SGD':
    results_save_path = ('../results/Experiment1/{15}/{16}/'
                         'Results_CelebA_{14}/'
                         'input_dim_{5}/init_{6}/layers_{0}/neuron_{1}/'
                         'lambda_cvx_{10}_mean_{11}/optim_{8}lr_{2}momen_{7}/'
                         'gen_{9}/batch_{3}/trial_{4}_last_{12}_qudr').format(
                                    args.NUM_LAYERS,
                                    args.NUM_NEURON,
                                    args.LR,
                                    args.BATCH_SIZE,
                                    args.TRIAL,
                                    args.INPUT_DIM,
                                    args.initialization,
                                    args.momentum,
                                    'SGD',
                                    args.N_GENERATOR_ITERS,
                                    args.LAMBDA_CVX,
                                    args.LAMBDA_MEAN,
                                    'full' if args.full_quadratic else 'inp',
                                    args.FEATURES,
                                    attribute,
                                    percentage)

elif args.optimizer == 'Adam':
    results_save_path = ('../results/Experiment1/{15}/{16}/'
                         'Results_CelebA_{14}/'
                         'input_dim_{5}/init_{6}/layers_{0}/neuron_{1}/'
                         'lambda_cvx_{11}_mean_{12}/'
                         'optim_{9}lr_{2}betas_{7}_{8}/gen_{10}/batch_{3}/'
                         'trial_{4}_last_{13}_qudr').format(
                                     args.NUM_LAYERS,
                                     args.NUM_NEURON,
                                     args.LR, args.BATCH_SIZE,
                                     args.TRIAL,
                                     args.INPUT_DIM,
                                     args.initialization,
                                     args.beta1,
                                     args.beta2,
                                     'Adam',
                                     args.N_GENERATOR_ITERS,
                                     args.LAMBDA_CVX,
                                     args.LAMBDA_MEAN,
                                     'full' if args.full_quadratic else 'inp',
                                     args.FEATURES,
                                     attribute,
                                     percentage)

model_save_path = results_save_path + '/storing_models'

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mps = False  #torch.backends.mps.is_available()

def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y


df = pd.read_csv(args.DATASET_X)
df["values_{}".format(args.FEATURES)] = [None]*len(df)

X_data = src.datasets.CelebA_Features(
                                args.DATASET_X,
                                "../data/{}".format(args.FEATURES))
train_loader = torch.utils.data.DataLoader(X_data,
                                           batch_size=1,
                                           shuffle=True)


Y_data = src.datasets.CelebA_Features_Kernel(
                    args.DATASET_Y,
                    "../data/{}".format(args.FEATURES),
                    scale=.001)


indices = random.sample(range(len(Y_data)), len(train_loader))
Y_subset = torch.utils.data.Subset(Y_data, indices)

Y_loader = torch.utils.data.DataLoader(Y_subset,
                                       batch_size=1)

convex_f = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
convex_f.load_state_dict(
    torch.load(model_save_path + '/convex_f_epoch_{}.pt'.format(args.epoch)))
convex_f.eval()

convex_g = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
convex_g.load_state_dict(
    torch.load(model_save_path + '/convex_g_epoch_{}.pt'.format(args.epoch)))
convex_g.eval()

if args.cuda:
    convex_f.cuda()
    convex_g.cuda()

elif args.mps:
    convex_f.to("mps")
    convex_g.to("mps")

sum_list = list()
norm_list = list()
ot_loss_list = list()
for batch, _, _, _ in Y_loader:

    if args.cuda:
        batch = batch.cuda()

    temp_sum = convex_g(batch).reshape(-1).item()
    temp_norm = 0.5*batch.pow(2).sum(dim=1).mean().item()
    batch.requires_grad = True

    g_of_y = convex_g(batch).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, batch, create_graph=True)[0]

    f_grad_g_y = convex_f(grad_g_of_y)
    dot_prod = (grad_g_of_y * batch).sum(dim=1)

    loss_g = f_grad_g_y.item() - dot_prod.item()

    ot_loss_list.append(loss_g)
    sum_list.append(temp_sum)
    norm_list.append(temp_norm)

ot_loss_average = np.array(ot_loss_list).mean()
g_average = np.array(sum_list).mean()
norm_average = np.array(norm_list).mean()

f_list = list()
norm_x_list = list()
for imgs, ids, _, _ in train_loader:
    ids = ids.item()
    if args.cuda:
        imgs = imgs.cuda()
    elif args.mps:
        imgs = imgs.to("mps")

    val = (0.5*imgs.pow(2).sum(dim=1).mean().item() -
           convex_f(imgs).item())

    f_list.append(convex_f(imgs).item())
    norm_x_list.append(0.5*imgs.pow(2).sum(dim=1).mean().item())
    df.loc[ids, "values_{}".format(args.FEATURES)] = val

df["values_{}".format(args.FEATURES)] += (norm_average +
                                          ot_loss_average)

df.to_csv(args.DATASET_X, index=False)

# =============================================================================
# img_ids = df.sort_values(by="values", ascending=False)["image_id"][:36]
# array_img_vectors = np.array(
#     [skimage.io.imread("../data/celeba/Img_folder/Img/" + file)
#      for file in img_ids])
# 
# path = results_save_path+'/grid_epoch_{}_female.jpeg'.format(args.epoch)
# save_images_as_grid(path, array_img_vectors)
# 
# img_ids = df.sort_values(by="values1", ascending=False)["image_id"][:36]
# array_img_vectors = np.array(
#     [skimage.io.imread("../data/celeba/Img_folder/Img/" + file)
#      for file in img_ids])
# 
# 
# path = results_save_path+'/grid_epoch_{}_female_value2.jpeg'.format(args.epoch)
# save_images_as_grid(path, array_img_vectors)
# 
# =============================================================================
# =============================================================================
# 
# ##################################################################
# # cluster the top 10% images
# last_decile = df[df["values"] >= np.percentile(df["values"], 90)]
# 
# last_decile = last_decile.reset_index()
# X_data = src.datasets.CelebA(None,
#                              "../data/celeba/Img_folder/Img",
#                              df=last_decile,
#                              transform=transform)
# 
# train_loader = torch.utils.data.DataLoader(X_data,
#                                            batch_size=args.BATCH_SIZE)
# 
# space = []
# for imgs, ids, _ in train_loader:
#     if args.cuda:
#         imgs = imgs.cuda()
# 
#     with torch.no_grad():
#         features_vector = features(imgs).cpu().numpy()
# 
#     space.append(features_vector)
# 
# space = np.concatenate(space)
# 
# kmeans = KMeans(4)
# kmeans.fit(space)
# 
# last_decile["cluster"] = kmeans.labels_
# last_decile.to_csv("../data/celeba/celebA_female_last_decile.csv", index=False)
# 
# =============================================================================
