import numpy as np
import pandas as pd
import random

# set random seeds
seed = 10

np.random.seed(seed)
random.seed(seed)

def select_uniform_sample(df, size=9000):
    """
    Function that samples uniformly from the data.
    """

    # select sample
    sample = df.sample(size)

    path = "../data/celeba/experiment1_uniform_sample.csv"
    sample.to_csv(path)


def select_biased_sample(df,
                         size=9000,
                         feature="Wearing_Necktie",
                         percentage = .9):
    """
    Function that assembles a biased split of a dataframe df with respect to
    a certain feature.
    """

    # select features
    df_feature = df[(df[feature] == 1)]
    df_no_feature = df[(df[feature] == -1)]

    to_concat = [
            df_feature.sample(round(percentage*size)),
            df_no_feature.sample(round((1-percentage)*size))
            ]

    final = pd.concat(to_concat)

    path = "../data/celeba/experiment1_biased_sample_{}_{}.csv".format(
                                                            feature,
                                                            int(percentage*100)
                                                            )
    final.to_csv(path)

if __name__ == "__main__":
    # path to CelebA attributes' .csv
    csv_path = "../data/celeba/list_attr_celeba.csv"

    # load .csv as a dataframe
    df = pd.read_csv(csv_path, index_col=None)

    # select and save the data
    select_uniform_sample(df)
    for percentage in [.9, .6, .3, .1]:
        select_biased_sample(df,
                             size = 9000,
                             feature="Eyeglasses",
                             percentage=percentage)
        select_biased_sample(df,
                             size=9000,
                             feature = "Wearing_Necktie",
                             percentage = percentage)
        select_biased_sample(df,
                             size=9000,
                             feature="Wearing_Hat",
                             percentage=percentage)

        select_biased_sample(df,
                             size=9000,
                             feature="Smiling",
                             percentage=percentage)
