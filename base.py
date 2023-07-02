import functools
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

HIDDEN_SIZE = 5


def calculate_matrix_power(A, x: Union[int, float]):
    """ Calculates A^x like we learned in class"""
    U, S, V = torch.svd(A)
    return U @ torch.diag_embed(torch.pow(S, x)) @ V.T


def generate_data(n_samples=100, input_size=1):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random X values
    X = np.random.uniform(low=0, high=10, size=(n_samples, input_size))

    # Generate y values with some random noise
    y = 3 * np.average(X, 1, keepdims=True) + 2 + np.random.normal(scale=3, size=(n_samples, 1))

    if input_size == 1:
        # Plot the data
        plt.scatter(X, y, s=10)
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Synthetic Regression Dataset')
        plt.savefig('Synthetic Regression Dataset')

    # Convert the dataset to tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    return X_tensor, y_tensor, X, y


# Linear neural network with depth 2
class LinearNetDepth2(nn.Module):
    def __init__(self, input_size, output_size, is_bias=True):
        super(LinearNetDepth2, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE, bias=is_bias)
        self.fc2 = nn.Linear(HIDDEN_SIZE, output_size, bias=is_bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Linear neural network with depth 3
class LinearNetDepth3(nn.Module):
    def __init__(self, input_size, output_size, is_bias=True):
        super(LinearNetDepth3, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE, bias=is_bias)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=is_bias)
        self.fc3 = nn.Linear(HIDDEN_SIZE, output_size, bias=is_bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
