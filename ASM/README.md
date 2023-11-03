#ASM

## What is ASM?

ASM was introduced by Cootes and Taylor and is grounded on the concept of iteratively updating the shape of an object within an image to better align with the actual object's shape. It does so by using a set of predefined landmarks on the object that are statistically modelled from a training set of correctly annotated images.

For faces, these landmarks typically represent key facial features such as the eyes, nose, mouth, and jawline. The ASM uses a combination of these landmarks to define a 'mean shape,' which can be transformed to fit new instances of faces in new images.

## facial_landmark_asm_analysis.py 
This code is for training an ASM model using the public datasets. Try it out to understand the training process step by step. 

## Facial Landmark Detection with Active Shape Models

This Python project implements an Active Shape Model (ASM) for facial landmark detection. It involves processing images, detecting facial structures, and analyzing shape variations using techniques like Principal Component Analysis (PCA) and Procrustes analysis.

## Features

- Detection of facial landmarks in images.
- Alignment of shapes using Generalised Procrustes Analysis.
- Modelling of face shape variations using PCA.
- Visualisation tools for displaying facial landmarks and the mean face model.
- Utilities to fit the ASM to new images and refine landmark positions.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- Python 3.6 or newer
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Scikit-image (`skimage`)
- Scipy
- Dlib
- Pillow (PIL fork)

You can install these with the following command:

```bash
pip install numpy opencv-python matplotlib scikit-image scipy dlib pillow

## Data Directory

This directory is intended for storing datasets used in the project. To replicate the results of the analysis, you should download the necessary datasets and organise them as described below.

## Dataset Organization

The project expects the following directory structure for datasets:


Data/
├── 300W/
│ ├── Train/
│ │ ├── 300W_train.txt
│ │ └── images/
│ └── Test/
│ ├── 300W_test.txt
│ └── images/
└── FRGC/
├── Train/
│ ├── FRGC_train.txt
│ └── images/
└── Test/
├── FRGC_test.txt
└── images



## Downloading the Datasets
Datasets like 300W and FRGC can be found in the following - 
[https://github.com/jiankangdeng/MenpoBenchmark]
Additional datasets that can be used are COFW, MultiPIE, XM2VTS and FRGC