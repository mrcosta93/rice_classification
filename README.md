# Rice Type Classification with PyTorch

This project implements a simple feedforward neural network using PyTorch to perform binary classification on a rice dataset. The primary objective is to demonstrate a clean and structured deep learning workflow rather than build a highly optimized production model.

## Project Objective

The goal is to:

- Implement a neural network from scratch using PyTorch
- Structure a complete training pipeline
- Monitor loss and accuracy during training
- Evaluate model behavior on validation and test splits

This project focuses on implementation clarity and training mechanics.

## Dataset

The dataset used is Rice Type Classification, available on Kaggle:

- 10 numerical features extracted from rice grain images

- Binary target variable representing rice type

The dataset is already preprocessed according to its documentation. No extensive exploratory data analysis (EDA) or anomaly treatment was performed, as the focus is on the modeling process.

## Model Architecture

The neural network consists of:

- Input layer: 10 neurons (one per feature)

- Hidden layer: 10 neurons (fully connected)

- Output layer: 1 neuron

- Activation function: Sigmoid (for binary probability output)

Architecture summary:
```
Linear(10 → 10)
Linear(10 → 1)
Sigmoid
```
Total trainable parameters: 121

The final sigmoid activation constrains the output to the range (0, 1), representing the probability of belonging to the positive class.

## Data Processing

- Features are normalized using Min-Max scaling

- Dataset split:

  - 70% Training

  - 15% Validation

  - 15% Test

- Custom Dataset class implemented

- Data loaded using PyTorch DataLoader

- Batch size: 8

- Shuffling enabled during training

## Training Configuration

- Loss Function: Binary Cross-Entropy (BCELoss)

- Optimizer: Adam

- Learning Rate: 1e-3

- Training performed over multiple epochs

- Metrics tracked:

  - Training loss

  - Validation loss

  - Training accuracy

  - Validation accuracy

## Results

The model converges rapidly:

- Accuracy increases significantly within the first few epochs

- Loss decreases consistently

- Final training accuracy approaches ~99%

Due to the simplicity of the architecture and the dataset characteristics, convergence is fast and stable.

## Key Concepts Demonstrated

- Device management (CPU/GPU with CUDA)

- Custom Dataset implementation

- Mini-batch training

- Forward and backward propagation

- Optimizer step mechanics

- Model evaluation pipeline

- Monitoring overfitting behavior

## How to Run

Clone the repository:
```
git clone <repository-url>
cd <repository-folder>
```

Install dependencies:
```
pip install torch torchvision scikit-learn pandas matplotlib
```

Run the notebook:
```
jupyter notebook main.ipynb
```
## Technologies Used

- Python

- PyTorch

- NumPy

- Pandas

- Scikit-learn

- Matplotlib

## Author

Marcos Costa
Data Scientist
LinkedIn: https://www.linkedin.com/in/marcosdatadev
