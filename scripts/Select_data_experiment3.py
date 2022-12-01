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
                feature2="Wearing_Necktie"):
    """
    This function takes a dataframe as an input and sample four datasets where
    feature1 == 1 and respectively 90%, 60%, 30% and 10% of rows with
    feature1 == 1 have also feature2 == 1.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    feature1 : str, optional
        Sensitive attribute. The default is "Male".
    feature2 : str, optional
        Other attribute. The default is "Wearing_Necktie".

    Returns
    -------
    None.

    """

    # select features
    df_male_no_tie = df[(df[feature1] == 1) & (df[feature2] == -1)]
    df_male_tie = df[(df[feature1] == 1) & (df[feature2] == 1)]

    df_female = df[(df[feature1] == -1)]

    len_male_tie = len(df_male_tie)

    for percentage in [.9, .6, .3, .1]:

        to_concat_male = [
            df_male_tie.sample(round(percentage*len_male_tie)),
            df_male_no_tie.sample(round((1-percentage)*len_male_tie))
            ]

        final_male = pd.concat(to_concat_male)

        path = "../data/celeba/experiment3_Male_{}_{}.csv".format(
                                                            feature2,
                                                            int(percentage*100)
                                                            )
        final_male.to_csv(path)

    final_female = df_female.sample(len_male_tie)

    path = "../data/celeba/experiment3_Female_{}.csv".format(feature2)
    final_female.to_csv(path)


def select_data_female(df,
                       feature1="Male",
                       feature2="Wearing_Necktie"):
    """
    This function takes a dataframe as an input and sample four datastes where
    feature1 == -1 and respectively 90%, 60%, 30% and 10% of rows with
    feature1 == -1 have also feature2 == 1.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    feature1 : str, optional
        Sensitive attribute. The default is "Male".
    feature2 : str, optional
        Other attribute. The default is "Wearing_Necktie".

    Returns
    -------
    None.

    """

    # select features
    df_female_no_tie = df[(df[feature1] == -1) & (df[feature2] == -1)]
    df_female_tie = df[(df[feature1] == -1) & (df[feature2] == 1)]

    df_male = df[(df[feature1] == 1)]

    len_female_tie = len(df_female_tie)

    for percentage in [.9, .6, .3, .1]:

        to_concat_female = [
            df_female_tie.sample(round(percentage*len_female_tie)),
            df_female_no_tie.sample(round((1-percentage)*len_female_tie))
            ]

        final_female = pd.concat(to_concat_female)

        path = "../data/celeba/experiment3_Female_{}_{}.csv".format(
                                                            feature2,
                                                            int(percentage*100)
                                                            )
        final_female.to_csv(path)

    final_male = df_male.sample(len_female_tie)

    path = "../data/celeba/experiment3_Male_{}.csv".format(feature2)
    final_male.to_csv(path)


# path to CelebA attributes' .csv
csv_path = "../data/celeba/list_attr_celeba.csv"
partition_path = "../data/celeba/list_eval_partition.csv"

# load .csv as dataframes
df = pd.read_csv(csv_path, index_col=None)
df_part = pd.read_csv(partition_path)

# we merge with the partition dataset to keep the test set outside the samples
new_df = pd.merge(df, df_part, how="left", on="image_id")

# select and save the data
select_data(new_df[new_df["partition"]==0])
select_data(new_df[new_df["partition"]==0],
                   feature2="Eyeglasses")
select_data(new_df[new_df["partition"]==0],
                   feature2="Wearing_Hat")
select_data(new_df[new_df["partition"]==0],
                   feature2="Smiling")

path_test = "../data/celeba/experiment3_test.csv"
new_df[new_df["partition"]==1].to_csv(path_test)