import argparse
import torch
import numpy as np
import pandas as pd
import json

from hydra import initialize, compose
from pathlib import Path

from src.evaluation import load_iccns, compute_optimal_transport_map
from src.utils import set_random_seeds
from src.datasets import BiasedMNIST_Features, BiasedMNIST_Features_Kernel

def compute_Kantorovich_potential(x, convex_f):
    return (.5*x.pow(2).sum() - convex_f(x)).item()

def compute_OT_loss(X_data, Y_data, convex_f, convex_g, cuda = True):

    Y_loader = torch.utils.data.DataLoader(Y_data,
                                       batch_size=1)
    OT_loss = list()
    for batch, _, _, _ in Y_loader:

        if cuda:
            batch = batch.cuda()

        batch.requires_grad = True

        grad_g_of_batch = compute_optimal_transport_map(batch, convex_g)

        f_grad_g_batch = convex_f(grad_g_of_batch)
        dot_prod = (grad_g_of_batch * batch).sum()

        loss_g = f_grad_g_batch - dot_prod
        OT_loss.append(loss_g.item() + .5 * batch.pow(2).sum().item())

    X_loader = torch.utils.data.DataLoader(X_data,
                                           batch_size=1,
                                           shuffle=True)
    f_values = list()
    for batch, _, _ in X_loader:

        if cuda:
            batch = batch.cuda()

        f_values.append(.5 * batch.pow(2).sum().item() - convex_f(batch).item())

    return np.array(OT_loss).mean() + np.array(f_values).mean()

def main():
    with initialize(version_base=None,
                    config_path="../results/training/{}/{}/.hydra".format(args.dataset, args.config)):

        # load config
        cfg = compose(config_name="config")

        set_random_seeds(cfg.settings.seed)

        cuda = not cfg.settings.no_cuda and torch.cuda.is_available()

        # directories
        root = Path("../")
        data_dir = root / "data"
        biased_mnist = data_dir / "biased_mnist"
        split_dir = biased_mnist / "resnet18_features/full_{}".format(cfg.data.bias)
        features_dir = split_dir / "trainval"

        # data
        X_data = BiasedMNIST_Features(
            root=data_dir,
            bias=cfg.data.bias,
            split="train")

        Y_data = BiasedMNIST_Features_Kernel(
            root=data_dir,
            bias=cfg.data.bias,
            split="test",
            var=cfg.data.kernel_variance)

        # results save path
        results_save_path = "../results/training/{}/{}".format(args.dataset, args.config)

        # load iccns
        convex_f, convex_g = load_iccns(results_save_path,
                                        cfg,
                                        args.epoch)


        wasserstein = compute_OT_loss(X_data, Y_data, convex_f, convex_g, cuda)

        tsv_path = "../results/Experiment1/wasserstein_biased_mnist.tsv"
        with open(tsv_path, "a") as f:
            f.write(cfg.data.bias + "\t" +
                    str(wasserstein) + "\n")

        # indexes traning set
        with open(biased_mnist / "train_ixs.json", "r") as file:
            indices = json.load(file)

        # dataframe
        df_X = pd.DataFrame(columns=["index", "KP"])

        for i in range(len(indices)):
            x = torch.load(features_dir / "{}.pt".format(indices[i]))
            df_X.loc[i, ["index", "KP"]] = indices[i], compute_Kantorovich_potential(x, convex_f)

        df_X.to_csv("../results/Experiment1/KP_biased_mnist_{}.csv".format(cfg.data.bias), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiment evaluation')

    parser.add_argument('--config',
                        type=str,
                        default='2023-07-09/22-50-20',
                        help='configuration to use in the experiment')

    parser.add_argument('--dataset',
                        type=str,
                        default='biased_mnist',
                        help='dataset used in the experiment')

    parser.add_argument('--epoch',
                        type=int,
                        default=25,
                        help='epoch checkpoint to load for the experiment')

    args = parser.parse_args()

    main()
