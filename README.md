# Breast Cancer Classification using Logistic Regression and Neural Networks

This project uses machine learning to classify breast cancer tumours as either malignant or benign using the Breast Cancer Wisconsin dataset from `scikit-learn`.

The aim of the project was to compare a logistic regression baseline model with a feed-forward neural network built in PyTorch, and to evaluate how well each model generalises to unseen data.

## Project Overview

Breast cancer classification is a binary classification problem. Each tumour sample is described by 30 numerical features, including measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry.

The project involved:

- Loading and exploring the dataset
- Visualising the class distribution
- Using PCA to visualise the data in two dimensions
- Splitting the data into training and test sets
- Standardising the input features
- Training a logistic regression baseline model
- Building and training a neural network in PyTorch
- Testing weight decay as a regularisation method
- Using 5-fold cross-validation to evaluate generalisation

## Dataset

The dataset used was the Breast Cancer Wisconsin dataset from `scikit-learn`.

It contains:

- 569 samples
- 30 numerical features
- 2 classes: malignant and benign

In `scikit-learn`, the target labels are coded as:

- `0` = malignant
- `1` = benign

## Models

### Logistic Regression Baseline

A logistic regression model was used as a baseline because it is simple, interpretable and suitable for binary classification.

### Neural Network

A feed-forward neural network was implemented using PyTorch.

The architecture was:

- Input layer: 30 features
- Hidden layer: 16 neurons
- Activation function: ReLU
- Output layer: 1 logit
- Loss function: `BCEWithLogitsLoss`
- Optimiser: Adam

I also tested weight decay to reduce overfitting and improve generalisation.

## Results

| Model | Train Loss | Test Loss | Train Accuracy | Test Accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression Baseline | 0.3262 | 0.3116 | 92.09% | 94.73% |
| Neural Network | 0.0577 | 0.0588 | 98.46% | 98.25% |
| Neural Network + Weight Decay | 0.0533 | 0.0546 | 98.46% | 99.12% |

To get a more reliable estimate of performance, I also used 5-fold cross-validation.

| Metric | Mean Value | Standard Deviation |
|---|---:|---:|
| Train Loss | 0.0508 | 0.01 |
| Test Loss | 0.0820 | 0.03 |
| Train Accuracy | 98.55% | 0.26 |
| Test Accuracy | 97.19% | 1.02 |

Although the single train-test split achieved 99.12% test accuracy with weight decay, the cross-validation result of 97.19% gives a more realistic estimate of generalisation.

## Key Findings

The neural network outperformed the logistic regression baseline, achieving lower loss and higher accuracy.

Weight decay slightly improved test performance, suggesting that regularisation helped the model generalise better.

Cross-validation showed that the single test split may have slightly overestimated performance, highlighting the importance of evaluating models across multiple data splits.

## Skills Demonstrated

- Python
- PyTorch
- scikit-learn
- NumPy
- Matplotlib
- Logistic regression
- Neural networks
- Binary classification
- Feature standardisation
- PCA
- Cross-validation
- Model evaluation
- Regularisation

## Files in this Repository

```text
breast-cancer-classification/
│
├── README.md
├── breast_cancer_classification.ipynb
├── Machine_learning_project_write_up.pdf
├── requirements.txt
└── images/
