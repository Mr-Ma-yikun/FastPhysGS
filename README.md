# FastPhysGS: Efficient and Robust Physical 3DGS-MPM Simulation via Interior Filling and Text-Physics Optimization (NeurIPS 2026)

## Introduction

This repository contains the official PyTorch implementation for the NeurIPS 2026 paper titled "Efficient and Robust Physical 3DGS-MPM Simulation via Interior Filling and Text-Physics Optimization" by Yikun Ma, Yiqing Li, Jingwen Ye, Zhongkai Wu, Weidong Zhang, Lin Gao and Zhi Jin.

## Abstract

Extending 3D Gaussian Splatting (3DGS) to 4D physical simulation remains challenging. Based on the Material Point Method (MPM), existing methods either rely on manual parameter tuning or distill dynamics from video diffusion models, limiting the generalization and optimization efficiency. Recent attempts using LLMs/VLMs suffer from a text/image-to-3D perceptual gap, yielding unstable physics behavior. In addition, they often ignore the surface structure of 3DGS, leading to implausible motion. We propose FastPhysGS, a fast and robust framework for physics-based dynamic 3DGS simulation: (1) Instance-aware Particle Filling (IPF) with Monte Carlo Importance Sampling (MCIS) to efficiently populate interior particles while preserving geometric fidelity; (2) Bidirectional Graph Decoupling Optimization (BGDO), an adaptive strategy that rapidly optimizes material parameters predicted from a VLM. Experiments show FastPhysGS achieves high-fidelity physical simulation in 1 minute using only 7 GB runtime memory, outperforming prior works with broad potential applications.

## Installation

1. Create a conda environment:

```
conda create -n fastphysgs python=3.11.9
conda activate fastphysgs
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e third_party/gaussian-splatting/submodules/simple-knn/
```

2. We use 3D Gaussian Splatting https://github.com/graphdeco-inria/gaussian-splatting as the scene representation. Please follow its guide to reconstruct your own scenes, or directly use the public 3DGS datasets adopted by PhysGaussian / Physics3D.

## Demo

FastPhysGS directly takes a trained 3DGS scene as input. The data folder is organized as follows, where `point_cloud.ply` follows the original 3DGS format:

```
dataset/
└── pillow2sofa/
    └── point_cloud/
        └── iteration_30000/
            └── point_cloud.ply
```

Then set the scene path of the config, e.g. `train.model_path: 'dataset/pillow2sofa'`. 


Put a 3DGS checkpoint into `dataset/`, and run the following code to quickly reproduce our results.

```
python main.py --config configs/pillow2sofa.yaml --tag pillow2sofa --gpu 0
```

The rendered frames and the result video are exported to `outputs/pillow2sofa/images/` and `outputs/pillow2sofa/videos/`.


## Citation

If you find this work helpful, please consider citing:

```
@inproceedings{ma2026fastphysgs,
  title     = {Efficient and Robust Physical 3DGS-MPM Simulation via Interior Filling and Text-Physics Optimization},
  author    = {Ma, Yikun and Li, Yiqing and Ye, Jingwen and Wu, Zhongkai and Zhang, Weidong and Gao, Lin and Jin, Zhi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

Feel free to reach out for any questions or issues related to the project. Thank you for your interest!
