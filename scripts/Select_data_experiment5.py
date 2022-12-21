import numpy as np
import pandas as pd
import random
# set random seeds
seed = 10

np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    # path to CelebA attributes' .csv
    csv_path = "../data/celeba/list_attr_celeba.csv"

    # load .csv as a dataframe
    df = pd.read_csv(csv_path, index_col=None)

    # select and save the data
    female = df[df["Male"] == -1]
    male = df[df["Male"] == 1]

    female.to_csv("../data/celeba/celeba_female.csv")
    male.to_csv("../data/celeba/celeba_male.csv")