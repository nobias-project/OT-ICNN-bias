# Studying Bias in Visual Features through the Lens of Optimal Transport
This project aims at exploiting Optimal Transport techniques for the purpose of bias detection in image data. The repository contains the code for the paper *"Studying Bias in Visual Features through the Lens of Optimal Transport"* (currently under-review at Data Mining and Knowledge Discovery) to ensure the reproducibility of its results. 

This work has been carried out in the context of the European Training Network [NoBIAS](https://nobias-project.eu).

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
4. Download [Biased MNIST](https://github.com/erobic/occam-nets-v1) dataset data and place it in the folder ```./OT-ICNN-bias/data/biased_mnist```.


5. Extract features from CelebA's images by running:
```console
cd scripts
python ./feature_extraction.py dataset="celeba"
```
or
```console
cd scripts
python ./feature_extraction.py dataset="biased_mnist"
```

Note that we use [Hydra](https://hydra.cc/docs/intro/) configuration manager, you can override some of the configurations by running:
```console
python ./feature_extraction.py features="resnet50"
```

## Run the Experiments
### Baseline comparison
6. Create toy datasets for this experiments by running:
```console
python ./Select_toy_datasets.py
```

7. Train the networks to compute the quadratic Wasserstein distance.
```console
python ./Toy_datasets_training.py
```

8. Run the notebook ```../notebook/Evaluation_Toy_Data.ipynb```

### Wasserstein distance for bias detection
#### CelebA
9. Select suitable splits of CelebA data by running:
```console
python ./Select_data_celebA.py
```

10. Train the networks to compute the quadratic Wasserstein distance.
```console
python ./celeba_training.py
```
11. Evaluate through 
```console
python ./experiment_eval_celeba.py
```
12. Run ```../notebook/Compute_Wasserstein.ipynb```
biased MNIST
#### CelebA

13. Train the networks to compute the quadratic Wasserstein distance.
```console
python ./biased_mnist_training.py
```

14. Evaluate through 
```console
python ./experiment_eval_biased_mnist.py
```

15. Run ```../notebook/Compute_Wasserstein.ipynb```

### Impact of Dimensionality Reduction

16. Run the code
```console
python ./dimensionality_reduction_CelebA_splits.py --method="PCA" --dimension="3"
```
You can  override the configurations to use other methods or dimensions.

17. Train using
18. Evaluate

### Case Study
19.  Select Data
20. reduce
21. train 
22. evaluate


## Acknowledgments
#### :eu: This work has received funding from the European Union’s Horizon 2020 research and innovation programme under Marie Sklodowska-Curie Actions (grant agreement number 860630) for the project ‘’NoBIAS - Artificial Intelligence without Bias.

## References 
```
@InProceedings{pmlr-v119-makkuva20a,
  title = 	 {Optimal transport mapping via input convex neural networks},
  author =       {Makkuva, Ashok and Taghvaei, Amirhossein and Oh, Sewoong and Lee, Jason},
  booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
  pages = 	 {6672--6681},
  year = 	 {2020},
  editor = 	 {III, Hal Daumé and Singh, Aarti},
  volume = 	 {119},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--18 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf},
  url = 	 {https://proceedings.mlr.press/v119/makkuva20a.html},
  abstract = 	 {In this paper, we present a novel and principled approach to learn the optimal transport between two distributions, from samples. Guided by the optimal transport theory, we learn the optimal Kantorovich potential which induces the optimal transport map. This involves learning two convex functions, by solving a novel minimax optimization. Building upon recent advances in the field of input convex neural networks, we propose a new framework to estimate the optimal transport mapping as the gradient of a convex function that is trained via minimax optimization. Numerical experiments confirm the accuracy of the learned transport map. Our approach can be readily used to train a deep generative model. When trained between a simple distribution in the latent space and a target distribution, the learned optimal transport map acts as a deep generative model. Although scaling this to a large dataset is challenging, we demonstrate two important strengths over standard adversarial training: robustness and discontinuity. As we seek the optimal transport, the learned generative model provides the same mapping regardless of how we initialize the neural networks. Further, a gradient of a neural network can easily represent discontinuous mappings, unlike standard neural networks that are constrained to be continuous. This allows the learned transport map to match any target distribution with many discontinuous supports and achieve sharp boundaries.}
}

@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}

@article{shrestha2022occamnets,
  title={OccamNets: Mitigating Dataset Bias by Favoring Simpler Hypotheses},
  author={Shrestha, Robik and Kafle, Kushal and Kanan, Christopher},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}

@Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
}
```
