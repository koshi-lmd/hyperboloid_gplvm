# hyperboloid_gplvm
This repository contains the official implementation for Hyperboloid GPLVM for Discovering Continuous Hierarchies via Nonparametric Estimation (AISTATS 2025, Accepted).

Hyperboloid GPLVM extends the GPLVM, sparse-GPLVM, and Bayesian GPLVM to the hyperbolic space. We develop hyperboloid kernel and optimization algorithms that follow previous research on kernel methods and optimization on Riemannian manifolds.

## Abstract
Dimensionality reduction (DR) offers interpretable representations of complex high-dimensional data. Recent DR methods leverage hyperbolic geometry to obtain faithful low-dimensional embedding of high-dimensional hierarchical relationships. However, existing approaches rely on neighbor embedding, frequently ruining the continuous nature of hierarchical structures. This paper presents hyperboloid Gaussian process (GP) latent variable models (hGP-LVMs) to embed high-dimensional hierarchical data while preserving the implicit continuity via nonparametric estimation. We adopt generative modeling using the GP, which brings effective hierarchical embedding and executes ill-posed hyperparameter tuning. This paper presents three variants that employ original point, sparse point, and Bayesian estimations. We establish their learning algorithms by incorporating the Riemannian optimization and active approximation scheme of GP-LVM. Furthermore, we employ the reparameterization trick for scalable learning of latent variables in the Bayesian estimation method. In the last part of this paper, we apply hGP-LVMs to several datasets and show their ability to represent high-dimensional hierarchies in low-dimensional spaces.

## BibTeX citation
If you use this code in your work, please cite our paper as follows (this information will be updated):
```bibtex
@article{watanabe2024hyperboloid,
  title={Hyperboloid GPLVM for Discovering Continuous Hierarchies via Nonparametric Estimation},
  author={Watanabe, Koshi and Maeda, Keisuke and Ogawa, Takahiro and Haseyama, Miki},
  journal={arXiv preprint arXiv:2410.16698},
  year={2024}
}
```

## Installation
We tested our code in the environment as follows:
- Ubuntu: 22.04
- python: 3.10.12
- numpy: 1.26.2
- scipy: 1.11.4
- tqdm: 4.66.1

## Note
- Our code does not depend on the GPU usage.
- We include the datasets used in our paper (dataset/). The results exhibited in our paper will be reproduced using 'demo.ipynb'.
- In the 'demo.ipynb', we partially used the original implementation of the [PoincareMaps](https://github.com/facebookresearch/PoincareMaps), [RoWN](https://github.com/ml-postech/RoWN), and [GPy library](https://gpy.readthedocs.io/en/deploy/). We appreciate their contributions.
