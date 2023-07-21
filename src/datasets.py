import os
import torch
import json
import pandas as pd
import numpy as np

from torch.utils import data
from skimage import io
from pathlib import Path

class Toy_Dataset(data.Dataset):

    def __init__(self, path, ground_truth = 1):
        super(Toy_Dataset, self).__init__()
        temp = np.load(path)
        temp = temp[temp[:,-1] == ground_truth]

        self.X = temp[:, :-1]
        self.y = temp[:,-1]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item):
        self.X[item]
        return torch.from_numpy(self.X[item]), self.y[item]

class CelebA(data.Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 df=None,
                 transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            df (pd.DataFrame): if csv_file is None, a panda DataFrame must be
            passed.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if csv_file:
            self.celeba = pd.read_csv(csv_file)
        else:
            self.celeba = df

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.celeba.loc[idx, "image_id"])
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return image, idx, self.celeba.loc[idx, "image_id"]


class CelebA_Features(data.Dataset):
    def __init__(self,
                 csv_file,
                 root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the features vectors.
        """
        self.celeba = pd.read_csv(csv_file)

        self.root_dir = root_dir

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                "{}.pt".format(
                                    self.celeba.loc[idx, "image_id"][:-4])
                                )
        x = torch.load(img_path)

        return (x.reshape(-1).detach(),
                idx,
                self.celeba.loc[idx, "image_id"],
                self.celeba.loc[idx, "Male"])


class CelebA_Features_Kernel(data.Dataset):
    def __init__(self,
                 csv_file,
                 root_dir,
                 var=0.1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the features vectors.
            var (float): standard deviation of the kernel distribution.
        """
        self.celeba = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.var = var

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                "{}.pt".format(
                                    self.celeba.loc[idx, "image_id"][:-4])
                                )
        x = torch.load(img_path)

        if self.var != 0:

            m = torch.distributions.MultivariateNormal(
                                            x.reshape(-1),
                                            self.var*torch.eye(x.shape[1]))

            sample = m.sample()
        else:
            sample = x.reshape(-1)

        return (sample.detach(),
                idx,
                x.detach(),
                self.celeba.loc[idx, "image_id"],
                self.celeba.loc[idx, "Male"])

class BiasedMNIST_Features(data.Dataset):
    def __init__(self,
                 root,
                 bias="0.9",
                 split="train"):

        """Biased MNIST dataset.

        :param root: str
            Root directory for the dataset.
        :param bias: str, default "0.9"
            Bias level in the dataset.
            Must be in ["0.1", "0.5", "0.75", "0.9", "0.95", "0.99"].
        :param split: str, default "train"
            It loads the train, validation or test split.
        """

        # directories
        self._base_dir = Path(root) / "biased_mnist"
        self._features_dir = self._base_dir / "resnet18_features"

        # check that bias is in the correct range
        if bias not in ("0.1", "0.5", "0.75", "0.9", "0.95", "0.99"):
            raise ValueError("the argument bias is not in "
                             '["0.1", "0.5", "0.75", "0.9",'
                             ' "0.95", "0.99"]')

        # load images paths
        self.split = split

        if split not in ["train", "validation", "test"]:
            raise ValueError("the argument split is not in "
                             '["train", "validation", "test"]')
        elif split == "train":
            self._attr_dir = self._base_dir / "full_{}".format(bias)
            self._data_dir = self._features_dir / "full_{}".format(bias)
            self._images_dir = self._data_dir / "trainval"

            with open(self._base_dir / "train_ixs.json", "r") as file:
                indices = json.load(file)

        elif split == "validation":
            self._attr_dir = self._base_dir / "full_{}".format(bias)
            self._data_dir = self._features_dir / "full_{}".format(bias)
            self._images_dir = self._data_dir / "trainval"

            with open(self._base_dir / "val_ixs.json", "r") as file:
                indices = json.load(file)

        else:
            self._attr_dir = self._base_dir / "full"
            self._data_dir = self._features_dir / "full"
            self._images_dir = self._data_dir / "test"

            indices = [int(file.split(".")[0]) \
                       for file in os.listdir(self._images_dir) \
                       if file.split(".")[1] == "pt"]

        # load attributes
        attr = self._extract_attributes()
        self._attributes = [attr[i] for i in indices]

        self._images = [self._images_dir / "{}.pt".format(i) for i in indices]

    def _extract_attributes(self):
        if self.split == "test":
            path = self._attr_dir / "test.json"
        else:
            path = self._attr_dir / "trainval.json"

        with open(path, "r") as file:
            trainval = json.load(file)
            attributes = {}
            for element in trainval:
                if element["index"] not in attributes:
                    attributes[element["index"]] = element

        return attributes

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self._images[idx]
        x = torch.load(img_path)

        return (x.reshape(-1).detach(),
                idx,
                self._attributes[idx])
class BiasedMNIST_Features_Kernel(data.Dataset):
    def __init__(self,
                 root,
                 bias="0.9",
                 split="train",
                 var=0.1):

        """Biased MNIST dataset.

        :param root: str
            Root directory for the dataset.
        :param bias: str, default "0.9"
            Bias level in the dataset.
            Must be in ["0.1", "0.5", "0.75", "0.9", "0.95", "0.99"].
        :param split: str, default "train"
            It loads the train, validation or test split.
        :param var: float, default 0.1
            Variance of the kernel density estimator.
        """
        # var
        self.var = var

        # directories
        self._base_dir = Path(root) / "biased_mnist"
        self._features_dir = self._base_dir / "resnet18_features"

        # check that bias is in the correct range
        if bias not in ("0.1", "0.5", "0.75", "0.9", "0.95", "0.99"):
            raise ValueError("the argument bias is not in "
                             '["0.1", "0.5", "0.75", "0.9",'
                             ' "0.95", "0.99"]')

        # load images paths
        self.split = split

        if split not in ["train", "validation", "test"]:
            raise ValueError("the argument split is not in "
                             '["train", "validation", "test"]')
        elif split == "train":
            self._attr_dir = self._base_dir / "full_{}".format(bias)
            self._data_dir = self._features_dir / "full_{}".format(bias)
            self._images_dir = self._data_dir/ "trainval"

            with open(self._base_dir / "train_ixs.json", "r") as file:
                indices = json.load(file)

        elif split == "validation":
            self._attr_dir = self._base_dir / "full_{}".format(bias)
            self._data_dir = self._features_dir / "full_{}".format(bias)
            self._images_dir = self._data_dir / "trainval"

            with open(self._base_dir / "val_ixs.json", "r") as file:
                indices = json.load(file)

        else:
            self._attr_dir = self._base_dir / "full"
            self._data_dir = self._features_dir / "full"
            self._images_dir = self._data_dir / "test"

            indices = [int(file.split(".")[0]) \
                       for file in os.listdir(self._images_dir) \
                       if file.split(".")[1] == "pt"]

        # load attributes
        attr = self._extract_attributes()
        self._attributes = [attr[i] for i in indices]

        self._images = [self._images_dir / "{}.pt".format(i) for i in indices]

    def _extract_attributes(self):
        if self.split == "test":
            path = self._attr_dir / "test.json"
        else:
            path = self._attr_dir / "trainval.json"

        with open(path, "r") as file:
            trainval = json.load(file)
            attributes = {}
            for element in trainval:
                if element["index"] not in attributes:
                    attributes[element["index"]] = element

        return attributes

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self._images[idx]
        x = torch.load(img_path)

        if self.var != 0:

            m = torch.distributions.MultivariateNormal(
                x.reshape(-1),
                self.var * torch.eye(x.shape[1]))

            sample = m.sample()
        else:
            sample = x.reshape(-1)

        return (sample.detach(),
                idx,
                x.detach(),
                self._attributes[idx])
