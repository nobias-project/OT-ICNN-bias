from __future__ import print_function
import argparse
import random
from src.optimal_transport_modules.icnn_modules import *
import numpy as np
import pandas as pd
import torch.utils.data
import src.datasets

# argument parsing
parser = argparse.ArgumentParser(description='Experiment1 Evaluation')

parser.add_argument('--DATASET_X',
                    type=str,
                    default=("../data/celeba/"
                             "experiment1_Male_Wearing_Necktie_90%.csv"),
                    help='X data')

parser.add_argument('--DATASET_Y',
                    type=str,
                    default=("../data/celeba/"
                             "experiment1_Female_Wearing_Necktie.csv"),
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
                    default=20,
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

parser.add_argument('--VARIANCE',
                    type=float,
                    default=0.001,
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
args.mps = False  # torch.backends.mps.is_available()


def compute_optimal_transport_map(y, convex_g):

    g_of_y = convex_g(y).sum()

    grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

    return grad_g_of_y

# load data
df = pd.read_csv(args.DATASET_Y)
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
                    var=args.VARIANCE)


indices = random.sample(range(len(Y_data)), len(train_loader))
Y_subset = torch.utils.data.Subset(Y_data, indices)

Y_loader = torch.utils.data.DataLoader(Y_subset,
                                       batch_size=1)

# load kantorovich potentials
convex_f = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
                                                        args.NUM_NEURON,
                                                        args.activation)
convex_f.load_state_dict(
    torch.load(model_save_path + '/convex_f_epoch_{}.pt'.format(args.epoch)))
convex_f.eval()

convex_g = Simple_Feedforward_4Layer_ICNN_LastInp_Quadratic(
                                                        args.INPUT_DIM,
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

# compute ot loss for g and aware norm of vectors in Y
wasserstein = list()

for batch, ids, _, _ in train_loader:

    if args.cuda:
        batch = batch.cuda()

    df.loc[ids, "values_{}".format(args.FEATURES)] = .5*batch.pow(2).sum().item() - convex_f(batch).item()

df.to_csv(args.DATASET_X, index=False)
