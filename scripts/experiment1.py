import argparse
import torch
import numpy as np
import pandas as pd

from hydra import initialize, compose

from src.evaluation import load_iccns, load_data
from src.evaluation import compute_OT_loss

def compute_Kantorovich_potential(x, convex_f):
    return (.5*x.pow(2).sum() - convex_f(x)).item()

def main():
    with initialize(version_base=None,
                    config_path="../scripts/outputs/{}/.hydra".format(args.config)):

        # load config
        cfg = compose(config_name="config")
        cuda = not cfg.settings.no_cuda and torch.cuda.is_available()

        # data
        X_data, Y_data = load_data(cfg)

        # load dataframe
        df_X = pd.read_csv(cfg.data.dataset_x)

        # load iccns
        convex_f, convex_g = load_iccns(cfg, args.epoch)


        wasserstein = compute_OT_loss(X_data, Y_data, convex_f, convex_g, cuda)
        with open("../results/experiment1_wasserstein.tsv", "a") as f:
            f.write(cfg.data.dataset_x + "\t" +
                    cfg.data.dataset_y + "\t" +
                    str(wasserstein) + "\n")

        X_loader = torch.utils.data.DataLoader(X_data,
                                           batch_size=1,
                                           shuffle=True)
        df_X["Kantorovich_potential"] = [np.NaN]*len(df_X)
        for x, id, _, _ in  X_loader:
            df_X.loc[id, "Kantorovich_potential"] = compute_Kantorovich_potential(x, convex_f)

        df_X.to_csv(cfg.data.dataset_x, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiment1')

    parser.add_argument('--config',
                        type=str,
                        default='',
                        help='configuration to use in the experiment')
    parser.add_argument('--epoch',
                        type=int,
                        default=1,
                        help='epoch checkpoint to load for the experiment')

    args = parser.parse_args()

    main()
