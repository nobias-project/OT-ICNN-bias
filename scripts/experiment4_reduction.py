import argparse
import os
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, MDS

def main():

    root = Path("../data")
    celeba = root/"celeba"

    uniform = "experiment1_uniform_sample.csv"
    unif_df = pd.read_csv(celeba/uniform)
    unif_lst = [torch.load(celeba/"resnet18"/"{}.pt".format(p[:-4])).detach().numpy() for p in unif_df.image_id]
    datasets = ["experiment1_biased_sample_Wearing_Hat_{}.csv".format(perc) for perc in [10,30,60,90]]

    for perc in [10, 30, 60, 90]:
        dataset = "experiment1_biased_sample_Wearing_Hat_{}.csv".format(perc)

        if args.method == "PCA":
            method = PCA(n_components=args.dimension)
        elif args.method == "TSNE":
            method = TSNE(n_components=args.dimension)
        elif args.method == "Isomap":
            method = Isomap(n_components=args.dimension,
                            metric="euclidean")
        elif args.method == "SpectralEmbedding":
            method = SpectralEmbedding(n_components=args.dimension)
        elif args.method == "MDS":
            method = MDS(n_components=args.dimension)
        else:
            raise NotImplementedError("Not implemented for this method")

        df = pd.read_csv(celeba/dataset)
        lst = [torch.load(celeba/"resnet18"/"{}.pt".format(p[:-4])).detach().numpy() for p in df.image_id]

        arr = np.concatenate(unif_lst + lst)
        files = list(unif_df.image_id) + list(df.image_id)

        reduced = method.fit_transform(arr)

        path = Path(celeba/"Wearing_Hat_{}_red_{}_{}".format(perc, args.method, args.dimension))
        os.makedirs(path, exist_ok=True)

        for file, row in zip(files, reduced):
            torch.save(torch.Tensor(row).reshape(1, -1), path/"{}.pt".format(file[:-4]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dimensionality reduction')

    parser.add_argument('--method',
                        type=str,
                        default="TSNE",
                        help='dimensionality reduction method')

    parser.add_argument('--dimension',
                        type=int,
                        default=3,
                        help='dimension')

    args = parser.parse_args()

    main()