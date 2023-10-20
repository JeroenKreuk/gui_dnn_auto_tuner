# gui_dnn_auto_tuner
DNN Auto Tuner is a graphical user interface (GUI) application developed in Python using the Tkinter library. It allows users to perform various tasks related to data preprocessing, feature selection, and hyperparameter tuning for training deep neural networks.
## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Features

- Open and process CSV database files.
- Dummyfication of categorical variables.
- Feature importance calculation using a RandomForestRegressor.
- Feature selection based on importance thresholds.
- SMOTE (Synthetic Minority Over-sampling Technique) and scaling options.
- Hyperparameter optimization using Optuna.
- Train and evaluate a neural network with optimized hyperparameters.
- Display results including accuracy, F1 score, precision, recall, and ROC AUC.
![image](https://github.com/JeroenKreuk/gui_dnn_auto_tuner/assets/85551796/16a8c736-7cfd-48dd-be4b-339eccde6f7a)

## Requirements

- Python 3.7 or higher
- Libraries: tkinter, pandas, scikit-learn, imbalanced-learn, optuna, tensorflow, and more (see the code for details)

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/JeroenKreuk/DNN-Manual-Tuner.git
