import argparse
import torch
import numpy as np
import pandas as pd

from hydra import initialize, compose

from src.evaluation import load_iccns, load_data
from src.evaluation import compute_OT_loss
from src.utils import set_random_seeds
from src.datasets import Toy_Dataset

def compute_Kantorovich_potential(x, convex_f):
    return (.5*x.pow(2).sum() - convex_f(x)).item()

def main():
    with initialize(version_base=None,
                    config_path="../results/training/{}/{}/.hydra".format(args.dataset, args.config)):

        set_random_seeds(cfg.settings.seed)

        # load config
        cfg = compose(config_name="config")
        cuda = not cfg.settings.no_cuda and torch.cuda.is_available()

        # data
        if args.experiment != 3:
            X_data, Y_data = load_data(cfg)

            # load dataframe
            df_X = pd.read_csv(cfg.data.dataset_x)

        else:
            X_data = Toy_Dataset(
                cfg.data.dataset,
                ground_truth=0)

            Y_data = Toy_Dataset(
                cfg.data.dataset,
                ground_truth=1)

        # results save path
        results_save_path = "../results/training/{}/{}".format(args.dataset, args.config)

        # load iccns
        convex_f, convex_g = load_iccns(results_save_path,
                                        cfg,
                                        args.epoch)


        wasserstein = compute_OT_loss(X_data, Y_data, convex_f, convex_g, cuda)

        tsv_path = "../results/Experiment{}/wasserstein.tsv".format(args.experiment)
        with open(tsv_path, "a") as f:
            f.write(cfg.data.dataset_x + "\t" +
                    cfg.data.dataset_y + "\t" +
                    str(wasserstein) + "\n")

        X_loader = torch.utils.data.DataLoader(X_data,
                                           batch_size=1,
                                           shuffle=True)
        if args.experiment != 3:
            df_X["Kantorovich_potential"] = [np.NaN]*len(df_X)
            for x, id, _, _ in  X_loader:
                df_X.loc[id, "Kantorovich_potential"] = compute_Kantorovich_potential(x, convex_f)

            df_X.to_csv(cfg.data.dataset_x, index=False)
            
        else:
            potentials = list()
            for x, _ in X_loader:
                potentials.append(compute_Kantorovich_potential(x, convex_f))

            np.save("../results/Experiment3/{}_potentials.npy".format(dataset), np.array(potentials))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiment evaluation')

    parser.add_argument('--config',
                        type=str,
                        default='',
                        help='configuration to use in the experiment')

    parser.add_argument('--dataset',
                        type=str,
                        default='celeba',
                        help='dataset used in the experiment')

    parser.add_argument('--experiment',
                        type=int,
                        default=1,
                        help='No of the experiment')

    parser.add_argument('--epoch',
                        type=int,
                        default=1,
                        help='epoch checkpoint to load for the experiment')

    args = parser.parse_args()

    main()
