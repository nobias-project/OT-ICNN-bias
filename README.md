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
Check the configurations at ```../scripts/config/feature_extraction_config.yaml```

## Run the Experiments
### Baseline comparison
6. Create two toy datasets (a pair of concentric circles and a pair of concentric circle plus a Gaussian) for this experiments by running:
```console
python ./Select_toy_datasets.py
```

7. Train the networks to compute the quadratic Wasserstein distance. 
```console
python ./Toy_datasets_training.py dataset="../data/toy/circles.npy"
```
and 

```console
python ./Toy_datasets_training.py dataset="../data/toy/circles_plus.npy"
```

To change other configurations check ```../scripts/config/toy_data_train_config.yaml``` 

8. Run the notebook ```../notebook/Evaluation_Toy_Data.ipynb```

### Wasserstein distance for bias detection
#### CelebA

9. Select suitable splits of CelebA data by running:
```console
python ./Select_data_celebA.py
```

10. Train the networks to compute the quadratic Wasserstein distance for the attribute Wearing_Necktie with 30% of bias wrt the uniform sample.
```console
python ./celeba_training.py   dataset_x="../data/celeba/experiment1_Male_Wearing_Necktie_30.csv"
  dataset_y="../data/celeba/experiment1_uniform_sample.csv"
```
The experiments where carried out for four different attributes (Wearing_Necktie, Wearing_Hat, Eyeglasses. and Smiling) and 4 different bias level (10%, 30%, 60%, 90%).

Check the other congigurations at ```../scripts/config/celeba_train_config.yaml```

11. Evaluate through 
```console
python ./experiment_eval_celeba.py --config="YYYY-MM-DD/HH-MM-SS" --epoch=25
```
YYYY-MM-DD/HH-MM-SS is the time stamp of the training as saved by Hydra.

12. Run ```../notebook/Compute_Wasserstein.ipynb```

#### Biased MNIST

13. Train the networks to compute the quadratic Wasserstein distance.
```console
python ./biased_mnist_training.py
```

14. Evaluate through 
```console
python ./experiment_eval_biased_mnist.py --config="YYYY-MM-DD/HH-MM-SS" --epoch=25
```

15. Run ```../notebook/Compute_Wasserstein.ipynb```

### Impact of Dimensionality Reduction

16. Run the code
```console
python ./dimensionality_reduction_CelebA_splits.py --method="PCA" --dimension="3"
```
You can  override the configurations to use other methods or dimensions. We experimented with PCA, TSNE, Isomap, and SpectralEmbedding. We reduce the space to 3, 50 and 150 dimensions (Note that for t-SNE we reduced only to 3 dimension because higher dimensions were too computational intensive).  

17. Train using

```console
python ./celeba_training.py   dataset_x="../data/celeba/experiment1_Male_Wearing_Necktie_30.csv"
  dataset_y="../data/celeba/experiment1_uniform_sample.csv" features = Wearing_Hat_reduced_10_PCA3
```

18. Evaluate as previous experiments and by running the notebook ```./notebooks/Evaluaton_Experiment_Dimensionality_Reduction.ipynb```. 

### Case Study
19.  Select data via
```console
python ./Select_data_case_study.py
```

20. Reduce dimension using
```console
python ./dimensionality_reduction_case_study.py
```
21. Train via 
```console
python ./celeba_training.py dataset_x = "../data/celeba/celeba_female.csv" dataset_y="../data/celeba/celeba_male.csv" features = "resnet18_reduced_PCA_3"
```
22. Evaluate as previous experiment and by running the notebook ```../notebooks/Evaluation_Case_Study.ipynb```


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
