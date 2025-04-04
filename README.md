# Source-Free Active Domain Adaptation via Influential-Points-Guided Progressive Teacher for Medical Image Segmentation

This repository contains the supported pytorch code and configuration files to reproduce of our method.

![](imgs/methods.png?raw=true)

# Introduction
This project propose a novel Source-Free Active Domain Adaptation (SFADA) method for medical image segmentation.  

We first introduce an influential point learning (IPL) slice-wise framework to actively select influential points for oracle annotation. The progressive teacher (ProT) model is designed to generate and utilize reliable pseudo-labels independently. The fully supervised learning is performed on labeled samples. For unlabeled samples, curriculum learning-based self-training is adopted to further reduce the negative impact of noisy pseudo-labels.

The experimental results on one multi-modal and two multi-center datasets demonstrated that our method outperformed state-of-the-art methods, even with only 2.5% of the labeling budget.
## Environment

Please prepare an environment with Python 3.7, Pytorch 1.7.1.  
`conda create --name SFADA --file requirements.txt`

## Dataset Preparation

Datasets can be acquired via following links:

- Multi-site Prostate Segmentation (PS) Dataset: click [here](https://liuquande.github.io/SAML/).
- Multi-centre, Multi-vendor and Multi-disease Cardiac Segmentation (M&MS) Dataset: click [here](https://www.ub.edu/mnms/).
- Multi-modal Brain Tumor (BraTS) Segmentation: click [here](https://www.med.upenn.edu/cbica/brats2020/data.html).

## Preprocess Data
Convert nii.gz Files to h5 Format to facilitate follow-up processing and training  
`python dataloaders/data_processing.py`

# Usage
### 1. Training a source model at a single center
`python train_source.py`

### 2. Select source-like samples in a target domain
`python selection/select_source_like_samples.py`

### 3. Select influential points using the IPL framework 
`python selection/select_influential_points.py`

### 4. Training a target model using the source model from step 1 
`python train_target.py`

### 5. Testing on the target domain
`python test.py`

### 6. Visualization results
`python inference.py`


# Results

![](imgs/Pro_result.png?raw=true)

![](imgs/heart_result.png?raw=true)

![](imgs/brats_result.png?raw=true)

## Acknowledge
Parts of codes are borrowed from [STDR](https://github.com/whq-xxh/SFADA-GTV-Seg).

