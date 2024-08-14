## Purpose
This repository was created to gain an understanding of deep learning and its techinques (e.g., training, data augmentation, k-folds cross validation, statistics for model analysis, etc.) by implementing the following:
- A multilayer perceptron (MLP) from scratch.
- A more complex MLP and a Convolutional Neural Network (CNN) using PyTorch

Details of the models and experiments are provided in a separate document.

## Reproducibility
To reproduce the project:
1. Ensure Python 3.10.14 is installed.
2. Set up a virtual environment using Python 3.10.14.
3. Run `pip install -r requirements.txt` to install the necessary packages.

## Project Structure
The project is organized as follows:

- **`config/`**: Contains the hyperparameters for each type of model
- **`data/`**: 
  - **`test_images/`**: Contains handwritten digits I drew for model inference tests
  - **`MNIST/`**: Contains the bytestream files of the MNIST dataset
- **`model/`**: Contains saved models with the following naming conventions:
  - MLP from scratch: `mlp-scratch-{testaccuracy}.pkl`
  - PyTorch CNN and MLP: `{model_type}-{prop_aug}aug{norm/aug-distribution-approach}-testaccuracy.pkl`
- **`src/`**: 
  - Contains the `custom_dataset` class, used for approach 1 of splitting the train dataset into normal and augmented handwritten digits (detailed in `model-reports.md`). This class was isolated in a Python script due to multiprocessing issues with the DataLoader when using Jupyter notebooks.
- **`notebooks/`**: Contains the model development process for each model created.
  - `mlp-model-selection.ipynb` and `cnn-model-selection.ipynb`: Include hyperparameter tuning, model training, and performance evaluation with model statistics using PyTorch.
  - `scratch-mlp-model-selection.ipynb`: Includes implementation of MLP and corresponding algorithms (e.g., Stochastic Gradient Descent) without deep learning libraries.

## Future Plans
- Improve the MLP created from scratch by implementing a dynamic computation graph that optimizes operations before execution (similar to PyTorch) and enabling support for L-layer architecture
- Develop a Convolutional Neural Network (CNN) from scratch


