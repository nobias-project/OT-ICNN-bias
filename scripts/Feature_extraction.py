import hydra
from omegaconf import DictConfig

import os
import torch
import torch.nn as nn
import pandas as pd

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from skimage import io
from PIL import Image

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

    if cuda:
        features.cuda()

    # load data
    if cfg.dataset == "celeba":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(160)])

        df = pd.read_csv("../data/celeba/list_attr_celeba.csv")
        root_dir = "../data/celeba/Img_folder/Img"
        os.makedirs("../data/{}/{}".format(cfg.dataset, cfg.features),
                    exist_ok=True)

        for i in df.index:
            img_path = os.path.join(root_dir,
                                    df.loc[i, "image_id"])

            image = io.imread(img_path)
            image = transform(image)

            if args.cuda:
                image.cuda()
            elif args.mps:
                image.to("mps")

            features_tensor = features(image.reshape(1, *image.shape))
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
                    image.cuda()

                features_tensor = features(image.reshape(1, *image.shape))
                save_path = "../data/{}/{}/{}.pt".format(cfg.dataset,
                                                         cfg.features,
                                                         name[:-4])
                torch.save(features_tensor, save_path)

if __name__ == "__main__":
    main()