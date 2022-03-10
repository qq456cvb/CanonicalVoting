<h1 align="center">
Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes
</h1>

<p align='center'>
<img align="center" src='images/intro.jpg' width='70%'> </img>
</p>

<div align="center">
<h3>
<a href="https://qq456cvb.github.io">Yang You</a>, Zelin Ye, Yujing Lou, Chengkun Li, Yong-Lu Li, Lizhuang Ma, Weiming Wang, Cewu Lu
<br>
<br>
CVPR 2022
<br>
<br>
<a href='https://arxiv.org/pdf/2011.12001.pdf'>
  <img src='https://img.shields.io/badge/Paper-PDF-orange?style=flat&logo=arxiv&logoColor=orange' alt='Paper PDF'>
</a>
<a href='https://qq456cvb.github.io/projects/canonical-voting'>
  <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=googlechrome&logoColor=green' alt='Project Page'>
</a>
  <!-- <a href='https://colab.research.google.com/'>
    <img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'>
  </a> -->
<br>
</h3>
</div>

Canonical Voting is a 3D detection method that disentangles the direct offset into Local Canonical Coordinates (LCC), box scales and box orientations. Only LCC and box scales are regressed while box orientations are generated by a canonical voting scheme. Finally, a LCC-aware back-projection checking algorithm iteratively cuts out bounding boxes from the generated vote maps, with the elimination of false positives. Our model achieves state-of-the-art performance on challenging large-scale datasets of real point cloud scans: ScanNet, SceneNN and SUN RGB-D.

# Contents
- [Overview](#overview)
- [Installation](#installation)
- [Train and Test on ScanNet](#train-and-test-on-scannet)
- [Test on SceneNN](#test-on-scenenn)
- [Train and Test on SUN RGB-D](#train-and-test-on-sun-rgb-d)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)

# Overview
This is the official Pytorch implementation of our work: Canonical Voting.
# Installation
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) v0.4.2
- Install our custom Hough Voting module under `houghvoting` folder, by running `python setup.py install`
- Tested with PyTorch v1.3.1 + CUDA 10.0
- Other dependecies: 
```
pip install hydra-core scipy scikit-learn tqdm shapely numpy-quaternion pickle plyfile
```

# Train and Test on ScanNet
<details>
<summary>Data Preparation</summary>

You will need to first download [ScanNet](https://github.com/ScanNet/ScanNet) original dataset and [Scan2CAD](https://github.com/skanti/Scan2CAD) labels with oriented bounding boxes. Then download Scan2CAD model segments [here](https://drive.google.com/drive/folders/1yKIcQuJte9vToRLbZYgwdYqUDECBYs1T?usp=sharing) and then download our preprocessed ground-truth boxes [here](https://drive.google.com/drive/folders/1i4ctu3oxwYG19kczqNgryj5uMnZVQZCv?usp=sharing) for evaluation. Adjust their path accordingly in `config/config.yaml`.
</details>

<details>
<summary>Start Training</summary>

To train model jointly for all categories, with one unified model:
```
python train_joint.py
```
To train model separately for each category:
```
python train_separate.py category=03211117,04379243,02808440,02747177,04256520,03001627,02933112,02871439,others -m
```
</details>

<details>
<summary>Evaluate mAP</summary>

Once trained, you can evaluate the model's mAP on ScanNet val set.

To eval the jointly trained model:
```
python eval_joint.py
```
To eval the separately trained model:
```
python eval_separate.py
```
</details>

# Test on SceneNN
<details>
<summary>Data Preparation</summary>

You will need to download our processed [SceneNN](https://mega.nz/folder/n7hzDQxb#mV8t4d7psPYN5bSkkxHuYw) data, which contains raw segmentation labels, instance labels and bounding box annotations. Set `scene_nn_root` in `config.yaml` to your downloaded directory.
</details>

<details>
<summary>Evaluate mAP</summary>

Run `eval_joint.py` or `eval_separate.py` with modified variable `SCENENN=True`.
</details>

# Train and Test on SUN RGB-D
Coming soon.

# Pretrained Models
<details>
<summary>Pretrained Model on ScanNet</summary>

Pretrained models for both joint and separate training settings can be found [here](https://drive.google.com/drive/folders/1Af5mRVwwI370txOREXkooea8nK_SwzGk?usp=sharing). You will get about 15.4 mAP and 21.7 mAP for joint and separate training settings, respectively.
</details>

<details>
<summary>Pretrained Model on SUN RGB-D</summary>
Coming soon.
</details>

# Citation
If you find our algorithm useful or use our processed data, please consider citing:
```
@article{you2022canonical,
  title={Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes},
  author={You, Yang and Ye, Zelin and Lou, Yujing and Li, Chengkun and Li, Yong-Lu and Ma, Lizhuang and Wang, Weiming and Lu, Cewu},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
