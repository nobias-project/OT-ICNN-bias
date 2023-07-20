# Optimal Transport for Bias Detection in Images
This project aims at exploiting Optimal Transport techniques for the purpose of bias detection in image data.

This work is inspired by the paper [optimal transport mapping via input convex neural neworks](https://arxiv.org/abs/1908.10962)

The repository is forked from [https://github.com/AmirTag/OT-ICNN](https://github.com/AmirTag/OT-ICNN)

## Get Started 
To use our code and reproduce our results please follow the following steps. 


1. Once you have forked and cloned the repository, you can create a conda enviroment 

```console
conda env create -f environment.yml

conda activate ot-bias
```
2. install the package locally
```console
pip install -e .
```

3. Download [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) data and place it in the folder ```./OT-ICNN-bias/data/celeba``` or open a python console and run:
```python
>>> import torchvision
>>> torchvision.datasets.CelebA("./data", download=True)
```
Make sure that the structure of the directory ```data``` is the following:

```bash
data
└── celeba
    ├── img_align_celeba
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    .   .
    .   .
    .   .
    │   └── 202599.jpg
    ├── list_attr_celeba.csv
    ├── list_bbox_celeba.csv
    ├── list_eval_partition.csv
    └── list_landmarks_align_celeba.csv

```

4. Extract features from CelebA's images by running:
```console
cd scripts
python ./feature_extraction.py
```

Note that we use [Hydra](https://hydra.cc/docs/intro/) configuration manager, you can override some of the configurations by running:
```console
python ./feature_extraction.py features="resnet50"
```

5. Select suitable splits of CelebA data by running:
```console
python ./Select_data_experiment1.py
```

6. Train the ICCNs. The code saves results and checkpoints in the folder ```./results```
```console
python ./celeba_training.py
```

Again you can override the configurations, for example
```console
python ./celeba_training.py training.epochs=50 training.optimizer="RMSProp" settings.no_cuda=True

```

7. Hydra stores the configuration used for each run into folders named ```./results/training/DATASET_NAME/YYYY-MM-DD/HH-MM-SS/.hydra``` You can test the goodness of the training with your configuration by running pytest. 
You need to specify the epoch checkpoint you want to test as well.  
```console
cd ../tests

pytest --config='YYYY-MM-DD/HH-MM-SS/' --dataset='celeba' --epoch_to_test=40 --verbose
```


## Acknowledgments
```
* https://github.com/AmirTag/OT-ICNN
```

## References 
```
@Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
}
```
