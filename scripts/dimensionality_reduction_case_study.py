import argparse
import os
import torch
import numpy as np

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, MDS

from src.utils import set_random_seeds

def load_data_as_numpy(dataset, features):
    path = Path("../data/{}/{}".format(dataset, features))
    lst = [torch.load(path/f).detach().numpy() for f in os.listdir(path)]
    array = np.concatenate(lst)

    return os.listdir(path), array

def save_data_as_tensor(dataset, features, method, dimension, files, array):
    path = Path("../data/{}/{}_reduced_{}_{}".format(dataset, features, method, dimension))
    os.makedirs(path, exist_ok=True)

    for file, row in zip(files, array):
        torch.save(torch.Tensor(row).reshape(1,-1), path/file)

    return None


def main():

    # set random seeds
    set_random_seeds(98)

    files, space = load_data_as_numpy(args.dataset, args.features)

    if args.method == "PCA":
        method = PCA(n_components=args.dimension)
    elif args.method == "TSNE":
        method = TSNE(n_components=args.dimension)
    elif args.method == "Isomap":
        method = Isomap(n_components=args.dimension)
    elif args.method == "SpectralEmbedding":
        method = SpectralEmbedding(n_components=args.dimension)
    elif args.method == "MDS":
        method = MDS(n_components=args.dimension)
    else:
        raise NotImplementedError("Not implemented for this method")

    space = method.fit_transform(space)
    save_data_as_tensor(args.dataset,
                        args.features,
                        args.method,
                        args.dimension,
                        files,
                        space)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dimensionality reduction')

    parser.add_argument('--features',
                        type=str,
                        default='facenet',
                        help='features to be reduce')

    parser.add_argument('--dataset',
                        type=str,
                        default='celeba',
                        help='dataset')

    parser.add_argument('--method',
                        type=str,
                        default="PCA",
                        help='dimensionality reduction method')

    parser.add_argument('--dimension',
                        type=int,
                        default=3,
                        help='dimension')

    args = parser.parse_args()

    main()