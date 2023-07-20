import hydra
from omegaconf import DictConfig

import os
import shutil
import torch
import torch.nn as nn
import pandas as pd

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from facenet_pytorch import InceptionResnetV1
from skimage import io
from PIL import Image
from src.models import AE

@hydra.main(version_base=None, config_path="config", config_name="feature_extraction_config")
def main(cfg: DictConfig):

    cuda = not cfg.no_cuda and torch.cuda.is_available()

    # load features extractor
    if cfg.features == "resnet18":
        features = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        features.fc = nn.Identity()

    elif cfg.features == "resnet50":
        features = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
        features.fc = nn.Identity()

    elif cfg.features == "facenet":
        features = InceptionResnetV1(pretrained='vggface2').eval()

    elif cfg.features == "autoencoder":
        path_ae = "../results/autoencoder/autoencoder_19.pth"
        features = AE(nc=3, ngf=128, ndf=128, latent_variable_size=512)
        features.load_state_dict(torch.load(path_ae))
        features.eval()


    if cuda:
        features = features.cuda()

    # load data
    if cfg.dataset == "celeba":
        if cfg.features == "autoencoder":
            transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(160)])

        df = pd.read_csv("../data/celeba/list_attr_celeba.csv")
        root_dir = "../data/celeba/img_align_celeba"
        os.makedirs("../data/{}/{}".format(cfg.dataset, cfg.features),
                    exist_ok=True)

        for i in df.index:
            img_path = os.path.join(root_dir,
                                    df.loc[i, "image_id"])

            image = io.imread(img_path)
            image = transform(image)

            if cuda:
                image = image.cuda()

            # compute features
            if cfg.features == "autoencoder":
                features_tensor = features.encode(image.reshape(1, *image.shape))
            else:
                features_tensor = features(image.reshape(1, *image.shape))

            # save features tensor
            save_path = "../data/{}/{}/{}.pt".format(cfg.dataset,
                                                     cfg.features,
                                                     df.loc[i, "image_id"][:-4])
            torch.save(features_tensor, save_path)


    if cfg.dataset == "oxford-iiit-pet":
        transform = transforms.Compose([transforms.ToTensor()])
                                        # transforms.Resize(160)])
        root_dir = "../data/oxford-iiit-pet/images"
        os.makedirs("../data/{}/{}".format(cfg.dataset, cfg.features),
                    exist_ok=True)

        images = os.listdir(root_dir)
        for name in images:
            if name[-4:] != ".mat":
                img_path = os.path.join(root_dir,
                                        name)

                image = Image.open(img_path)
                image = image.convert("RGB")
                image = transform(image)

                if cuda:
                    image = image.cuda()

                features_tensor = features(image.reshape(1, *image.shape))
                save_path = "../data/{}/{}/{}.pt".format(cfg.dataset,
                                                         cfg.features,
                                                         name[:-4])
                torch.save(features_tensor, save_path)

    if cfg.dataset == "biased-mnist":

        # load data
        if cfg.features == "resnet18":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(160)])
        else:
            raise(NotImplementedError)

        root_dir = "../data/biased_mnist"
        directories = ["full/test",
                       "full_0.1/trainval",
                       "full_0.5/trainval",
                       "full_0.75/trainval",
                       "full_0.9/trainval",
                       "full_0.95/trainval",
                       "full_0.99/trainval"]

        for dir in directories:
            src_dir = os.path.join(root_dir, dir)
            trg_dir = os.path.join(root_dir,
                                   "{}_features".format(cfg.features),
                                   dir)

            os.makedirs(trg_dir,
                        exist_ok=True)

            images = os.listdir(src_dir)
            for name in images:
                # compute features
                if name[-4:] == ".jpg":
                    img_path = os.path.join(src_dir,
                                            name)

                    image = Image.open(img_path)
                    image = image.convert("RGB")
                    image = transform(image)

                    if cuda:
                        image = image.cuda()

                    features_tensor = features(image.reshape(1, *image.shape))
                    save_path = os.path.join(trg_dir,
                                             "{}.pt".format(name[:-4]))
                    torch.save(features_tensor, save_path)

if __name__ == "__main__":
    main()