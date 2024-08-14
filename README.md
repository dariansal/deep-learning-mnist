## Purpose
This repository was created to gain an understanding of deep learning and its techinques (e.g., training, data augmentation, k-folds cross validation, statistics for model analysis, etc.) by implementing the following:
- A multilayer perceptron (MLP) from scratch.
- A more complex MLP and a Convolutional Neural Network (CNN) using PyTorch

Details of the models and experiments are provided in a separate document.

## Reproducibility
To reproduce the experiments:
1. Ensure Python 3.10.14 is installed.
2. Set up a virtual environment with Python 3.10.14.
3. Run `pip install -r requirements.txt` to install the necessary packages.

## Project Structure
The project is organized as follows:

- **`config/`**: Contains the hyperparameters for each type of model.
- **`data/`**: 
  - **`test_images/`**: Contains handwritten digits I drew for model inference tests.
  - **`MNIST/`**: Contains the bytestream files of the MNIST dataset.
- **`model/`**: Contains saved models with the following naming conventions:
  - MLP from scratch: `mlp-scratch-{testaccuracy}.pkl`
  - CNN and PyTorch MLP: `{model_type}-{prop_aug}aug{norm/aug-distribution-approach}-testaccuracy.pkl`
- **`src/`**: 
  - Contains the `custom_dataset` class, used for approach 1 of splitting the train/augmented dataset (detailed in document). This class was isolated in a Python script due to multiprocessing issues with the DataLoader when using Jupyter notebooks.
- **`notebooks/`**: Contains the following Jupyter notebooks:
  - `mlp-model-selection.ipynb`: Details the logic and code for obtaining the final MLP model (PyTorch).
  - `cnn-model-selection.ipynb`: Details the logic and code for obtaining the final CNN model (PyTorch).
  - `scratch-mlp-model-selection.ipynb`: Details the logic and code for obtaining the final MLP model (from scratch).

## Future Plans
- Improve the MLP created from scratch by implementing a dynamic computation graph that optimizes operations before execution (similar to PyTorch) and enabling support for L-layer architecture
- Develop a Convolutional Neural Network (CNN) from scratch


