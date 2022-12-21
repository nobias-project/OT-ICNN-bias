import os
import torch
import pandas as pd

from torch.utils import data
from skimage import io

class Toy_Dataset(nn.Module):

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


class Pet_Features(data.Dataset):
    def __init__(self,
                 root_dir,
                 cats=None):
        """
        Args:
            root_dir (string): Directory with all the features vectors.
            cats (bool): retreive only images with attribute cat = cats.
                        If None, ot retrieves all.
        """
        self.root_dir = root_dir

        if cats is None:
            self.list = os.listdir(root_dir)

        elif cats == 1:
            self.list = [file for file in os.listdir(root_dir)
                         if file[0].isupper()]
        elif cats == 0:
            self.list = [file for file in os.listdir(root_dir)
                         if not file[0].isupper()]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):

        def is_cat(name):
            if name[0].isupper():
                return 1
            return 0

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.list[idx])

        x = torch.load(img_path)

        return (x.reshape(-1).detach(),
                is_cat(self.list[idx]),
                self.list[idx])


class Pet_Features_Kernel(data.Dataset):
    def __init__(self,
                 root_dir,
                 cats=None,
                 var=0.1):
        """
        Args:
            root_dir (string): Directory with all the features vectors.
            cats (bool): retreive only images with attribute cat = cats.
                        If None, ot retrieves all.
            var (float): standard deviation of the kernel distribution.
        """
        self.root_dir = root_dir

        if cats is None:
            self.list = os.listdir(root_dir)

        elif cats == 1:
            self.list = [file for file in os.listdir(root_dir)
                         if file[0].isupper()]
        elif cats == 0:
            self.list = [file for file in os.listdir(root_dir)
                         if not file[0].isupper()]

        self.var = var

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):

        def is_cat(name):
            if name[0].isupper():
                return 1
            return 0

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.list[idx])

        x = torch.load(img_path)

        m = torch.distributions.MultivariateNormal(
                                        x.reshape(-1),
                                        self.var*torch.eye(x.shape[1]))
        sample = m.sample()

        return (sample.reshape(-1).detach(),
                is_cat(self.list[idx]),
                self.list[idx])
