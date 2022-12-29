import argparse
import os
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.evaluation import compute_OT_loss
from src.utils import set_random_seeds
from src.datasets import Toy_Dataset

def load_data_as_numpy(dataset, features):
    path = Path("../data/{}/{}".format(dataset, features))
    lst = [torch.load(path/f).detach().numpy() for f in os.listdir(path)]
    array = np.concatenate(lst)

    return os.listdir(path), array


def save_data_as_tensor(dataset, features, method, files, array):
    path = Path("../data/{}/{}_reduced_{}".format(dataset, features, method))
    os.makedirs(path, exist_ok=True)

    for file, row in zip(files, array):
        torch.save(torch.Tensor(row).reshape(1,-1), path/file)

    return None


def main():
    files, space = load_data_as_numpy(args.dataset, args.features)

    if args.method == "PCA":
        method = PCA(n_components=3)
    elif args.method == "TSNE":
        method = TSNE(n_components=3)
    else:
        raise NotImplementedError("Not implemented for this method")

    space = method.fit_transform(space)
    save_data_as_tensor(args.dataset,
                        args.features,
                        args.method,
                        files,
                        space)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dimensionality reduction')

    parser.add_argument('--features',
                        type=str,
                        default='resnet18',
                        help='features to be reduce')

    parser.add_argument('--dataset',
                        type=str,
                        default='celeba',
                        help='dataset')

    parser.add_argument('--method',
                        type=str,
                        default="TSNE",
                        help='dimensionality reduction method')

    args = parser.parse_args()

    main()