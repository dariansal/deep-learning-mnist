## Purpose
This repository was created to gain an understanding of deep learning and its techinques (e.g., training, data augmentation, k-folds cross validation, statistics for model analysis, etc.) by implementing the following:
- A multilayer perceptron (MLP) and its corresponding algorithms from scratch
- A more complex MLP and a Convolutional Neural Network (CNN) using PyTorch

## Reproducibility
To reproduce the project:
1. Ensure Python 3.10.14 is installed.
2. Set up a virtual environment using Python 3.10.14.
3. Run `pip install -r requirements.txt` to install the necessary packages.

## Project Structure
The project is organized as follows:

- **`config/`**: Contains the hyperparameters for each type of model
- **`data/`**: 
  - **`test_images/`**: Contains additional handwritten digits (drawn by myself) for model inference tests
  - **`MNIST/`**: Contains the bytestream files of the MNIST dataset
- **`model-reports/`**: Contains reports of the methodologies and analysis of each of the models
  - **`visuals/`** contains images that were present in the reports 
  - `pytorch-model-reports.md` contains analysis of the CNN and MLP implemented in PyTorch
  - `scratch-mlp-model-report.md` contains analysis of the MLP created from scratch
- **`models/`**: Contains saved models with the following naming conventions (placeholders enclosed in {}):
  - MLP from scratch: `mlp-scratch-{test_accuracy}.pkl`
  - PyTorch MLP: `mlp-{prop_aug}aug-{test_accuracy}.pkl`
  - PyTorch CNN: `cnn-{prop_aug}aug-{test_accuracy}.pkl`
- **`src/`**: 
  - `custom_dataset.py`: Contains the DataSetT class used for Approach 1 of splitting the train dataset into normalized/augmented proportions. Isolated class in a Python script due to multiprocessing issues with the DataLoader when using Jupyter notebooks. See "Data Augmentation Approaches" header below for more details.
- **`notebooks/`**: Contains the model development process for each model created.
  - `mlp-model-selection.ipynb` and `cnn-model-selection.ipynb` include hyperparameter tuning, model training, and performance evaluation with model statistics using PyTorch
  - `scratch-mlp-model-selection.ipynb`: Includes implementation of MLP and corresponding algorithms (e.g., Stochastic Gradient Descent) without deep learning libraries
- `notes.pdf`: Handwritten notes created to better understand and complete the project. Listed below are the most important sections to guide understanding.
  - Table of Contents (page 1)
  - The two approaches to splitting training data into normalized/augmented data for the PyTorch models (pages 7-8):
  - Derivation of backpropagation equations for the MLP model implemented from scratch (pages 18-20)

## Data Augmentation Approaches
Two different approaches were used for splitting the training data into normalized and augmented proportions:

 - Approach 1: Implemented in `mlp-model-selection.ipynb` using the DataSetT class in `src/custom_dataset.py`
 - Approach 2: Implemented in both `mlp-model-selection.ipynb` and `cnn-model-selection.ipynb`

Detailed explanations of the approaches can be found in both `model-reports/pytorch-model-reports.md` and `notes.pdf`.

## Summary of Results
Details of the models and their methodologies provided in `model-reports.md`. However, for a quick summary, the test results for the PyTorch MLP, PyTorch CNN, and the MLP from scratch are 99.02%, 99.56%, and 97.47% respectively.

## Future Plans
- Improve the MLP created from scratch by implementing a dynamic computation graph that optimizes operations before execution (similar to PyTorch) and enabling support for L-layer architecture
- Develop a Convolutional Neural Network (CNN) from scratch


