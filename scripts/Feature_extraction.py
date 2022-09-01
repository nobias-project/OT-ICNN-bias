import argparse
import os
import torch
import torch.nn as nn
import pandas as pd

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models import detection 
import facenet_pytorch as facenet
from skimage import io
from PIL import Image
from src import datasets

# Argument parsing
parser = argparse.ArgumentParser(description='Features Extraction')

parser.add_argument('--DATASET',
                    type=str,
                    default="oxford-iiit-pet",
                    help='dataset to extract the features from')

parser.add_argument('--FEATURES',
                    type=str,
                    default="resnet18",
                    help='Features extractor')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=True,
                    help='disables CUDA/mps training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mps = not args.no_cuda and torch.backends.mps.is_available()

# load features extractor
if args.FEATURES == "resnet18":
    features = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    features.fc = nn.Identity()

elif args.FEATURES == "resnet50":
    features = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    features.fc = nn.Identity()

elif args.FEATURES == "facenet":
    features = facenet.InceptionResnetV1(pretrained='vggface2').eval()

if args.cuda:
    features.cuda()
elif args.mps:
    features.to("mps")

# load data
if args.DATASET == "celeba":
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(160)])

    df = pd.read_csv("../data/celeba/list_attr_celeba.csv")
    root_dir = "../data/celeba/Img_folder/Img"
    os.makedirs("../data/{}/{}".format(args.DATASET, args.FEATURES),
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
        break
        save_path = "../data/{}/{}/{}.pt".format(args.DATASET,
                                                 args.FEATURES,
                                                 df.loc[i, "image_id"][:-4])
        torch.save(features_tensor, save_path)


if args.DATASET == "oxford-iiit-pet":
    transform = transforms.Compose([transforms.ToTensor()])
                                    # transforms.Resize(160)])
    root_dir = "../data/oxford-iiit-pet/images"
    os.makedirs("../data/{}/{}".format(args.DATASET, args.FEATURES),
                exist_ok=True)

    images = os.listdir(root_dir)
    for name in images:
        if name[-4:] != ".mat":
            img_path = os.path.join(root_dir,
                                    name)

            image = Image.open(img_path)
            image = image.convert("RGB")
            image = transform(image)

            if args.cuda:
                image.cuda()
            elif args.mps:
                image.to("mps")

            features_tensor = features(image.reshape(1, *image.shape))
            save_path = "../data/{}/{}/{}.pt".format(args.DATASET,
                                                     args.FEATURES,
                                                     name[:-4])
            torch.save(features_tensor, save_path)
    
    
# =============================================================================
# 
#     df = pd.read_csv("../data/celeba/list_attr_celeba.csv")
#     root_dir = "../data/celeba/Img_folder/Img"
#     os.makedirs("../data/{}/{}".format(args.DATASET, args.FEATURES),
#                 exist_ok=True)
# 
#     for i in df.index:
#         img_path = os.path.join(root_dir,
#                                 df.loc[i, "image_id"])
# 
#         image = io.imread(img_path)
#         image = transform(image)
# 
#         if args.cuda:
#             image.cuda()
#         elif args.mps:
#             image.to("mps")
# 
#         features_tensor = features(image.reshape(1, *image.shape))
#         break
#         save_path = "../data/{}/{}/{}.pt".format(args.DATASET,
#                                                  args.FEATURES,
#                                                  df.loc[i, "image_id"][:-4])
#         torch.save(features_tensor, save_path)
# 
# 
# =============================================================================
