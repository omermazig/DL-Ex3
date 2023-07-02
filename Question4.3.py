import copy
import functools

import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from base import LinearNetDepth2, LinearNetDepth3, generate_data, calculate_matrix_power
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing
import numpy as np


def relu_ntk(x, x_prime, device):
    mul = (x * x_prime).sum(-1)
    norm = x.norm(dim=-1) * x_prime.norm(dim=-1)
    norm_mul = torch.minimum(mul / norm, torch.ones(1).to(device))  # Xs are normalized. floating error gives 1+epsilon that killarccos
    return mul * (torch.pi - torch.arccos(norm_mul)) / torch.pi


def training_loop(model_trained, X, y, num_epochs, learning_rate, criterion):
    # Define the optimizer
    optimizer_train = optim.SGD(model_trained.parameters(), lr=learning_rate)

    # Training loop
    with torch.no_grad():
        u_t_empirical = [model_trained(X)]
        diffs_emp = [criterion(u_t_empirical[0], y)]

    for epoch in tqdm(range(num_epochs)):
        model_trained.train()

        pred = model_trained(X)
        loss = criterion(pred, y)

        # Backward pass and optimization
        optimizer_train.zero_grad()
        loss.backward()
        optimizer_train.step()

        u_t_empirical.append(pred)
        diffs_emp.append(loss)

    return u_t_empirical, diffs_emp



class ShallowActivatedNet(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(ShallowActivatedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False).to(device)
        torch.nn.init.normal_(self.fc1.weight)
        self.activation = nn.ReLU().to(device)
        self.accumulation = ((torch.randint(2, (1, hidden_size)) * 2 - 1) / np.sqrt(hidden_size)).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = torch.sum(self.accumulation * x, -1)
        return x


def main():
    torch.manual_seed(42)

    data = fetch_california_housing()
    print(data.feature_names)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    m = 100
    X, y = torch.tensor(data.data[:m]).to(device).float(), torch.tensor(data.target[:m]).to(device).float()
    X = torch.nn.functional.normalize(X, dim=-1)

    num_epochs = 10000
    lr = 0.0001
    H = relu_ntk(X.unsqueeze(0), X.unsqueeze(1), device)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for n, linestyle in zip([10, 50, 100], ['-', '--', ':']):
        model_trained = ShallowActivatedNet(X.shape[-1], n, device)
        with torch.no_grad():
            u_t_calc = model_trained(X)

        criterion = nn.MSELoss()
        u_t_trains, diffs_emp = training_loop(model_trained, X, y, num_epochs, lr, criterion)

        u_t_calcs = [copy.deepcopy(u_t_calc)]
        diffs_calc = [criterion(u_t_calc, y)]
        for epoch in range(num_epochs):
            u_dot = - H @ (u_t_calc - y)
            u_t_calc += lr * u_dot / m
            u_t_calcs.append(copy.deepcopy(u_t_calc))
            diffs_calc.append(criterion(u_t_calc, y))

        axs[0].plot([diff.detach().cpu() for diff in diffs_emp],
                 label=f"n={n}, empirical", color='blue', linestyle=linestyle)
        axs[0].plot([diff.detach().cpu() for diff in diffs_calc],
                 label=f"n={n}, calculated", color='orange', linestyle=linestyle)

        axs[1].plot([((u_t_calcs[j] - u_t_trains[j]).norm()
                      ).detach().cpu() for j in range(len(u_t_calcs))],
                 label=f"n={n}", color='green', linestyle=linestyle)

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Losses for Different Network Widths')
    axs[0].legend()
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('L2 Distance')
    axs[1].set_title('L2 Distance Between u_t Values')
    axs[1].legend()
    fig.show()


if __name__ == '__main__':
    main()
