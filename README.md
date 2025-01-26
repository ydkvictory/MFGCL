# MFGCL
# Identifying LncRNA-Disease Associations by Using Multi-modal Similarities Fusion and Graph Contrastive Learning

## Introduction
There is the source code of MFGCL (Identifying LncRNA-Disease Associations by Using Multi-modal Similarities Fusion and Graph Contrastive Learning)!

## Environmental Requirements
python 3.8.0</br>
torch 1.31.1</br>
torch-geometric 2.3.1</br>
scikit-learn 1.2.2</br>
numpy 1.21.0

## How to train model?
### 1. Processing data:
Please download our data zip and run `snf.py` to fuse similarity matrices of lncRNA and disease.
### 2. Sampling subgraphs:
The `util_mp.py` file provides the method for sampling subgraphs
### 3. Representation learning
The `model_test.py` file describes how to train model to learn representations of lncRNAs and diseases by contrasive learning and aggregate them. 
### 4. Associations predictions
By running `transclf.py` file, the model will be trained and the learning representations are fed into an MLP to predict potential associations.

## The flowchart of MFGCL
![1](https://github.com/user-attachments/assets/12af92f4-e5da-426c-81c8-c7a7a2569045)

## Notes
### Want to change datasets?
Please modify the defalut value of `--dataset` in the `transclf.py` file, for example 'dataset2', 'dataset3' or your own datasets. Meanwhile, you must update file paths used in the `load_data()` function in the `util_data.py` file.
