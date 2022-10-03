#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:55:17 2022

@author: simonefabrizzi
"""

from __future__ import print_function
import argparse
from src.optimal_transport_modules.icnn_modules import *
import torch.utils.data
import src.datasets
from src.utils import *
from torchvision.utils import make_grid
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Oxford Pet '
                                             'Experiment Evaluation')
parser.add_argument('--epoch',
                    type=int,
                    default=40,
                    metavar='S',
                    help='epoch to be evaluated')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA')

parser.add_argument('--BATCH_SIZE',
                    type=int,
                    default=10,
                    help='size of the batches')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mps = False  #torch.backends.mps.is_available()


def save_images_as_grid(path, array_img_vectors):

    array_img_vectors = torch.from_numpy(array_img_vectors)\
        .float().permute(0, 3, 1, 2)
    grid = make_grid(array_img_vectors, nrow=6, normalize=True)*255
    ndarr = grid.to('cpu', torch.uint8).numpy().T
    im = Image.fromarray(ndarr.transpose(1, 0, 2))

    im.save(path)


def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y


results_save_path = ("../results/Results_Pet_resnet18/input_dim_512/"
                     "init_trunc_inv_sqrt/layers_3/neuron_512/"
                     "lambda_cvx_0.5_mean_0.0/"
                     "optim_Adamlr_0.001betas_0.5_0.99/gen_5/batch_100/"
                     "trial_1_last_inp_qudr")
model_save_path = results_save_path + '/storing_models'


X_data = src.datasets.Pet_Features(
    "../data/oxford-iiit-pet/{}".format("resnet18"),
    cats=0)


Y_data = src.datasets.Pet_Features_Kernel(
    "../data/oxford-iiit-pet/{}".format("resnet18"),
    cats=1,
    scale=.0001)

train_loader = torch.utils.data.DataLoader(X_data,
                                           batch_size=1)

Y_loader = torch.utils.data.DataLoader(Y_data,
                                       batch_size=1)

convex_f = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(512,
                                                             512,
                                                             "leaky_relu")
convex_f.load_state_dict(
    torch.load(model_save_path + '/convex_f_epoch_{}.pt'.format(args.epoch)))
#convex_f = convex_f.eval()

convex_g = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(512,
                                                             512,
                                                             "leaky_relu")
convex_g.load_state_dict(
    torch.load(model_save_path + '/convex_g_epoch_{}.pt'.format(args.epoch)))
#convex_g = convex_g.eval()

if args.cuda:
    convex_f.cuda()

elif args.mps:
    convex_f.to("mps")

ot_loss_list = list()
sum_list = list()
norm_list = list()
for batch, _, _ in Y_loader:
    temp_sum = convex_g(batch).reshape(-1).sum().item()
    temp_norm = torch.linalg.norm(batch, 2, dim=1).pow(2).sum().item()

    batch.requires_grad = True

    g_of_y = convex_g(batch).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, batch, create_graph=True)[0]

    f_grad_g_y = convex_f(grad_g_of_y).mean()
    dot_prod = (grad_g_of_y * batch).sum(dim=1).mean()

    loss_g = f_grad_g_y - dot_prod

    ot_loss_list.append(loss_g.item())
    sum_list.append(temp_sum)
    norm_list.append(temp_norm)

ot_loss_average = sum(ot_loss_list)/len(Y_data)
g_average = sum(sum_list)/len(Y_data)
norm_average = sum(norm_list)/len(Y_data)

path_csv = ("../results/Results_Pet_resnet50/"
            "convex_f_{}_results(1).csv".format(args.epoch))

with open(path_csv, "w") as file:
    file.write("name,value_g_avg,value_ot_loss")
    for imgs, _, name in train_loader:

        if args.cuda:
            imgs = imgs.cuda()
        elif args.mps:
            imgs = imgs.to("mps")

        norm_square = (torch.linalg.norm(imgs.reshape(-1), 2).item())**2

        val = .5*norm_square - convex_f(imgs).item()

        val1 = val + (.5*norm_average + 110.13577758789066)
        val += (.5*norm_average - g_average)

        line = name[0] + "," + str(val) + "," + str(val1) + "\n"

        file.write(line)
