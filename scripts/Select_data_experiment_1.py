#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:45:33 2022

@author: simonefabrizzi
"""

import numpy as np
import pandas as pd
import random

# set random seeds
seed = 10

np.random.seed(seed)
random.seed(seed)


# function that assembles the data
def select_data(df,
                feature1="Male",
                feature2="Wearing_Hat"):

    # select features
    df_no_f1_f2 = df[(df[feature1] == -1) & (df[feature2] == 1)]
    df_no_f1_no_f2 = df[(df[feature1] == -1) & (df[feature2] == -1)]

    df_f1_no_f2 = df[(df[feature1] == 1) & (df[feature2] == -1)]
    df_f1_f2 = df[(df[feature1] == 1) & (df[feature2] == 1)]

    len_no_f1_f2 = len(df_no_f1_f2)

    for percentage in [.9, .6, .3, .1]:

        to_concat_female = [
            df_no_f1_f2.sample(round(percentage*len_no_f1_f2)),
            df_no_f1_no_f2.sample(round((1-percentage)*len_no_f1_f2))
            ]

        final_female = pd.concat(to_concat_female)

        path = "../data/celeba/experiment1_Female_{}_{}%.csv".format(
                                                            feature2,
                                                            int(percentage*100)
                                                            )
        final_female.to_csv(path)

    to_concat_male = [df_f1_no_f2.sample(round(.9*len_no_f1_f2)),
                      df_f1_f2.sample(round(.1*len_no_f1_f2))]

    final_male = pd.concat(to_concat_male)

    path = "../data/celeba/experiment1_Male_{}_{}%.csv".format(
                                                        feature2,
                                                        10
                                                        )
    final_male.to_csv(path)


# path to CelebA attributes' .csv
csv_path = "../data/celeba/list_attr_celeba.csv"

# load .csv as a dataframe
df = pd.read_csv(csv_path, index_col=None)

select_data(df)
select_data(df, feature2="Eyeglasses")
