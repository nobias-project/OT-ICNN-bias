from __future__ import print_function

import argparse
import torch
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import os
import logging
import torch.utils.data

import src.datasets
import src.optimal_transport_modules

from torch.autograd import Variable
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

from src.optimal_transport_modules.icnn_modules import *
from src.utils import *

# matplotlib.use('tkagg')


# Training settings. Important ones first
parser = argparse.ArgumentParser(description='Experiment1')

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

parser.add_argument('--epochs',
                    type=int,
                    default=40,
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

os.makedirs(model_save_path, exist_ok=True)

setup_logging(os.path.join(results_save_path, 'log.txt'))
results_file = os.path.join(results_save_path, 'results.%s')
results = ResultsLog(results_file % 'csv', results_file % 'html')
logging.info("saving to %s \n", results_save_path)
logging.debug("run arguments: %s", args)

kwargs = {'num_workers': 1, 'pin_memory': True} if (args.cuda or
                                                    args.mps) else {}

################################################################
# Data stuff

X_data = src.datasets.CelebA_Features(
                                args.DATASET_X,
                                "../data/{}".format(args.FEATURES))
train_loader = torch.utils.data.DataLoader(X_data,
                                           batch_size=args.BATCH_SIZE,
                                           shuffle=True,
                                           **kwargs)
logging.info("Created the data loader for X\n")


Y_data = src.datasets.CelebA_Features_Kernel(
                    args.DATASET_Y,
                    "../data/{}".format(args.FEATURES),
                    scale=.001)

############################################################
# Model stuff

# This loss is a relaxation of positive constraints on the weights
# Hence we penalize the negative ReLU


def compute_constraint_loss(list_of_params):

    loss_val = 0

    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val


def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

# Everything about model ends here
##############################################################


################################################################
# Everything related to both the convex functions

if args.NUM_LAYERS == 2:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_2Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
    else:
        convex_f = Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_2Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)

elif args.NUM_LAYERS == 3:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
    else:
        convex_f = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_3Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)

elif args.NUM_LAYERS == 4:
    
    if args.full_quadratic:
        convex_f = Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_4Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
    else:
        convex_f = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)

elif args.NUM_LAYERS == 5:

    if args.full_quadratic:
        convex_f = Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_5Layer_ICNN_LastFull_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
    else:
        convex_f = Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
        convex_g = Simple_Feedforward_5Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)

# Form a list of positive weight parameters
# and also initialize them with positive values
f_positive_params = []

for p in list(convex_f.parameters()):
    if hasattr(p, 'be_positive'):
        f_positive_params.append(p)

    p.data = torch.from_numpy(
        truncated_normal(p.shape,
                         threshold=1./np.sqrt(p.shape[1]
                                              if len(p.shape) > 1
                                              else p.shape[0]))).float()

g_positive_params = []

for p in list(convex_g.parameters()):
    if hasattr(p, 'be_positive'):
        g_positive_params.append(p)

    p.data = torch.from_numpy(
        truncated_normal(p.shape,
                         threshold=1./np.sqrt(p.shape[1]
                                              if len(p.shape) > 1
                                              else p.shape[0]))).float()

if args.cuda:
    convex_f.cuda()
    convex_g.cuda()

elif args.mps:
    convex_f.to("mps")
    convex_g.to("mps")

logging.info("Created and initialized the convex neural networks 'f' and 'g'")
num_parameters = sum([parameter.nelement()
                      for parameter in convex_f.parameters()])

logging.info("number of parameters: %d", num_parameters)

f_positive_constraint_loss = compute_constraint_loss(f_positive_params)

g_positive_constraint_loss = compute_constraint_loss(g_positive_params)

if args.optimizer == 'SGD':

    optimizer_f = optim.SGD(convex_f.parameters(),
                            lr=args.LR,
                            momentum=args.momentum)
    optimizer_g = optim.SGD(convex_g.parameters(),
                            lr=args.LR,
                            momentum=args.momentum)

if args.optimizer == 'Adam':

    optimizer_f = optim.Adam(convex_f.parameters(),
                             lr=args.LR,
                             betas=(args.beta1, args.beta2),
                             weight_decay=1e-5)
    optimizer_g = optim.Adam(convex_g.parameters(),
                             lr=args.LR,
                             betas=(args.beta1, args.beta2),
                             weight_decay=1e-5)

# Training stuff


def train(epoch):

    convex_f.train()
    convex_g.train()

    # count = 0

    w_2_loss_value_epoch = 0

    g_OT_loss_value_epoch = 0

    g_Constraint_loss_value_epoch = 0

    for batch_idx, (real_data, _, _, _) in enumerate(train_loader):

        if args.cuda:
            real_data = real_data.cuda()
        elif args.mps:
            real_data = real_data.to("mps")

        real_data = Variable(real_data)

        indices = random.sample(range(len(Y_data)), len(real_data))
        Y_subset = torch.utils.data.Subset(Y_data, indices)
        Y_loader = torch.utils.data.DataLoader(Y_subset,
                                               batch_size=len(real_data),
                                               shuffle=True,
                                               **kwargs)
        y, _, _, _ = next(iter(Y_loader))

        if args.cuda:
            y = y.cuda()
        elif args.mps:
            y = y.to("mps")

        y = Variable(y, requires_grad=True)

        optimizer_f.zero_grad()
        optimizer_g.zero_grad()

        g_OT_loss_val_batch = 0
        g_Constraint_loss_val_batch = 0

        # norm_g_parms_grad_full = 0

        for inner_iter in range(1, args.N_GENERATOR_ITERS+1):
            # First do a forward pass on y and compute grad_g_y
            # Then do a backward pass update on parameters of g

            optimizer_g.zero_grad()

            g_of_y = convex_g(y).sum()

            grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

            f_grad_g_y = convex_f(grad_g_of_y).mean()
            dot_prod = (grad_g_of_y * y).sum(dim=1).mean()

            loss_g = f_grad_g_y - dot_prod
            g_OT_loss_val_batch += loss_g.item()

            if args.LAMBDA_MEAN > 0:

                mean_difference_loss = args.LAMBDA_MEAN *\
                    (real_data.mean(0) - grad_g_of_y.mean(0)).pow(2).sum()
                variance_difference_loss = args.LAMBDA_MEAN *\
                    (real_data.std(0) - grad_g_of_y.std(0)).pow(2).sum()

                loss_g += mean_difference_loss + variance_difference_loss

            loss_g.backward()

            # Constraint loss for g parameters
            if args.LAMBDA_CVX > 0:
                g_positive_constraint_loss = args.LAMBDA_CVX *\
                    compute_constraint_loss(g_positive_params)
                g_Constraint_loss_val_batch += g_positive_constraint_loss
                g_positive_constraint_loss.backward()

            optimizer_g.step()

            # Maintaining the positive constraints on the convex_g_params
            if args.LAMBDA_CVX == 0:
                for p in g_positive_params:
                    p.data.copy_(torch.relu(p.data))

            # Just for the last iteration keep the gradient on f intact
            # otherwise need to do from scratch
            if inner_iter != args.N_GENERATOR_ITERS:
                optimizer_f.zero_grad()

        g_OT_loss_val_batch /= args.N_GENERATOR_ITERS
        g_Constraint_loss_val_batch /= args.N_GENERATOR_ITERS

        # norm_g_parms_grad_full /= args.N_GENERATOR_ITERS

        # Flip the gradient sign for parameters in convex_f
        # because we need to solve "sup" maximization for f
        for p in list(convex_f.parameters()):
            p.grad.copy_(-p.grad)

        remaining_f_loss = convex_f(real_data).mean()
        remaining_f_loss.backward()

        optimizer_f.step()

        # Maintain the "f" parameters positive
        for p in f_positive_params:
            p.data.copy_(torch.relu(p.data))

        w_2_loss_value_batch = (g_OT_loss_val_batch -
                                remaining_f_loss.item() +
                                0.5*real_data.pow(2).sum(dim=1).mean().item() +
                                0.5*y.pow(2).sum(dim=1).mean().item())
        w_2_loss_value_epoch += w_2_loss_value_batch

        g_OT_loss_value_epoch += g_OT_loss_val_batch
        g_Constraint_loss_value_epoch += g_Constraint_loss_val_batch

        if batch_idx % args.log_interval == 0:
            logging.info(('Train Epoch: {}'
                          ' [{}/{} ({:.0f}%)]'
                          ' g_OT_loss: {:.4f}'
                          ' g_Constraint_Loss: {:.4f}'
                          ' W_2_Loss: {:.4f} ').format(
                                        epoch,
                                        batch_idx * len(real_data),
                                        len(train_loader.dataset),
                                        100. * batch_idx / len(train_loader),
                                        g_OT_loss_val_batch,
                                        g_Constraint_loss_val_batch,
                                        w_2_loss_value_batch))

    w_2_loss_value_epoch /= len(train_loader)
    g_OT_loss_value_epoch /= len(train_loader)
    g_Constraint_loss_value_epoch /= len(train_loader)

    results.add(epoch=epoch,
                w2_loss_train_samples=w_2_loss_value_epoch,
                g_OT_train_loss=g_OT_loss_value_epoch,
                g_Constraint_loss=g_Constraint_loss_value_epoch)

    results.save()

    return (w_2_loss_value_epoch,
            g_OT_loss_value_epoch,
            g_Constraint_loss_value_epoch)

###################################################
# Training stuff


total_w_2_epoch_loss_list = []
total_g_OT_epoch_loss_list = []
total_g_Constraint_epoch_loss_list = []

for epoch in range(1, args.epochs + 1):

    # transported_y = compute_optimal_transport_map(y_plot, convex_g)

    # plot_transported_samples(transported_y, epoch)

    (w_2_loss_value_epoch,
     g_OT_loss_value_epoch,
     g_Constraint_loss_value_epoch) = train(epoch)

    total_w_2_epoch_loss_list.append(w_2_loss_value_epoch)
    total_g_OT_epoch_loss_list.append(g_OT_loss_value_epoch)
    total_g_Constraint_epoch_loss_list.append(g_Constraint_loss_value_epoch)

    if epoch % args.lr_schedule == 0:

        optimizer_g.param_groups[0]['lr'] *= 0.5

        optimizer_f.param_groups[0]['lr'] *= 0.5

    # if epoch % 10 == 0:
    if epoch % 1 == 0:
        torch.save(convex_f.state_dict(),
                   model_save_path + '/convex_f_epoch_{0}.pt'.format(epoch))
        torch.save(convex_g.state_dict(),
                   model_save_path + '/convex_g_epoch_{0}.pt'.format(epoch))

    else:
        torch.save(convex_f.state_dict(),
                   model_save_path + '/convex_f_lastepoch.pt')
        torch.save(convex_g.state_dict(),
                   model_save_path + '/convex_g_lastepoch.pt')

plt.plot(range(1, len(total_w_2_epoch_loss_list) + 1),
         total_w_2_epoch_loss_list,
         label='Training loss')

plt.xlabel('iterations')
plt.ylabel(r'$W_2$-loss value')
plt.savefig(results_save_path + '/training_loss.png')
plt.show()
plt.clf()

plt.plot(range(9, len(total_w_2_epoch_loss_list) + 1),
         total_w_2_epoch_loss_list[8:],
         label='Training loss')
plt.xlabel('iterations')
plt.ylabel(r'$W_2$-loss value')
plt.savefig(results_save_path + '/training_loss9+.png')
plt.show()

logging.info("Training is finished and the models"
             " and plots are saved. Good job :)")
