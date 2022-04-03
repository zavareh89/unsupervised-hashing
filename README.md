# Unsupervised Hashing algorithms in python


This repository contains the python implementation of some notable unsupervised hashing algorithms mostly based on spectral hashing formulation. If you found this code useful in your research, please consider citing one of the following papers:
 
```bib
@article{ESH,
title = {A non-alternating graph hashing algorithm for large-scale image search},
journal = {Computer Vision and Image Understanding},
pages = {103415},
year = {2022},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2022.103415},
url = {https://www.sciencedirect.com/science/article/pii/S1077314222000418},
author = {Sobhan Hemati and Mohammad Hadi Mehdizavareh and Shojaeddin Chenouri and Hamid R. Tizhoosh},
keywords = {Hashing, Graph hashing, Image search, Spectral hashing},
}
```

```bib
@INPROCEEDINGS{9506441,
  author    = {Hemati, Sobhan and Mehdizavareh, Mohammad Hadi and Babaie, Morteza and Kalra, Shivam and Tizhoosh, H.R.},
  booktitle = {2021 IEEE International Conference on Image Processing (ICIP)}, 
  title     = {A Simple Supervised Hashing Algorithm Using Projected Gradient and Oppositional Weights}, 
  year      = {2021},
  volume    = {},
  number    = {},
  pages     = {2748-2752},
  doi       = {10.1109/ICIP42928.2021.9506441}}
```

```bib
@article{HEMATI202244,
title = {Beyond neighbourhood-preserving transformations for quantization-based unsupervised hashing},
journal = {Pattern Recognition Letters},
volume = {153},
pages = {44-50},
year = {2022},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2021.11.007},
url = {https://www.sciencedirect.com/science/article/pii/S0167865521003974},
author = {Sobhan Hemati and H.R. Tizhoosh},
keywords = {Image Search, Unsupervised Hashing, Quantization, Binary Representation},
}
```

# What algorithms are available?
The following algorithms have been implemented:

1- [Iterative Quantization (ITQ)](https://ieeexplore.ieee.org/document/6296665) - See `demo_ITQ.py`

2- [Spectral Hashing (SH)](https://papers.nips.cc/paper/2008/hash/d58072be2820e8682c0a27c0518e805e-Abstract.html) - See `demo_SH.py`

3- [Kernelized Spectral Hashing (KSH)](https://www.ee.columbia.edu/ln/dvmm/publications/10/OKH_KDD2010.pdf) - See `demo_KSH.py`

4- [Discrete Graph Hashing (DGH)](https://papers.nips.cc/paper/2014/hash/f63f65b503e22cb970527f23c9ad7db1-Abstract.html) - See `demo_DGH.py`

5- [Large Graph Hashing with Spectral Rotation (LGHSR)](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14686/14394) also known as "Spectral Hashing with Spectral Rotation" (SHSR) - See `demo_LGHSR.py`

6- [Discrete Spectral Hashing (DSH)](https://dtaoo.github.io/papers/2019_DSH.pdf) - See `demo_DSH.py`

7- [Reversed Spectral Hashing (ReSH)](https://ieeexplore.ieee.org/document/7920418) - See `demo_ReSH.py`

Also we have proposed a new algorithm based on spectral hashing, called [Efficient Spectral Hashing (ESH)](). You can find the source code of this algorithm [here](https://github.com/sobhanhemati/Efficient-Spectral-Hashing-ESH-).

# Requirements
You should have `scipy`, `numpy`, and `scikit-learn`
installed. Also, for running ReSH algorithm, you need `tensorflow` 2.1 or higher.

# Usage
1- You should download one of the datasets in the [Datasets](#Datasets) section, put them in a folder (e.g. "C:\test") and set `dataset_name` parameter in the `demo` scripts to the dataset you downloaded. Also, you should change the `path` parameter to the dataset folder. For example if you want to use "labelme_vggfc7" dataset, you should change `dataset_name` and `path` parameters as follows:

```python
path = r'C:\test' # folder containing dataset
dataset_name = 'labelme_vggfc7'
```

2- `K` is number of bits.

# Datasets 
The following datasets are available:

[mnist_gist512](https://1drv.ms/u/s!Av1MQK8mV3J8gnkooTeL9ZdtCYtu)

[cifar10_gist512](https://1drv.ms/u/s!Av1MQK8mV3J8gnrlULhhHGy4Q88c)

[cifar10_vggfc7](https://www.dropbox.com/s/bnybq48ljtsyuit/cifar10_vggfc7.rar?dl=0)

[labelme_vggfc7](https://www.dropbox.com/s/0nc80qepzj8615f/labelme_vggfc7.rar?dl=0)

[nuswide_vgg](https://www.dropbox.com/s/6hl9t6oy78w028d/nuswide_vgg.rar?dl=0)

[colorectal_EfficientNet](https://www.dropbox.com/s/wdsalhu73bnrtsg/colorectal_EfficientNet.rar?dl=0)

[SUN397](http://www.mediafire.com/?790zq882c3j7d) (based on [SCQ paper](https://arxiv.org/abs/1802.06645))


# Some notes about our implementation
1- To reduce the computational complexity of calculating the affinity matrix in SH formulation, Most algorithms use the low-rank approximation proposed in [AGH paper](https://icml.cc/Conferences/2011/papers/6_icmlpaper.pdf). We also used it in all of the algorithms which are based on SH formulation (except ReSH). You can change the parameters related to this affinity approximation technique in "parameter initialization" cell in the `demo` scripts.

2- There are multiple ways to generate binary codes for unseen samples (samples which are not in training data). One technique is learning a linear projection matrix (or linear hash functions) using least-square estimation and projecting unseen samples onto this matrix. We have implemented this technique in `RRC` function (see `utilities` script). However, most papers have used another technique called out-of-sample extension proposed again by [AGH paper](https://icml.cc/Conferences/2011/papers/6_icmlpaper.pdf). We considered this technique as the default way for generating binary codes of unseen samples. 

3- You can compute affinity matrix for each dataset only once and save it. This reduces the time required for training multiple models on the same data (e.g. for different `K`s). We have implemented this in the `demo` scripts of SH-based methods (except ReSH).

4- For ReSH method, we couldn't get the same results with the paper. There are high chances that hyperparameters haven't been set properly in our implementation. Please consider this if you are going to use it and tune it in your case.


## Acknowledgements
We would like to thank the authors of [KNNH paper](https://github.com/HolmesShuan/K-Nearest-Neighbors-Hashing) for facilitating access to most datasets.
