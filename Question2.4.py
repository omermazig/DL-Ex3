import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from pyhessian import hessian
from torch.utils.data import TensorDataset, DataLoader

from base import generate_data, LinearNetDepth2, LinearNetDepth3, HIDDEN_SIZE


# Linear neural network with depth 4
class LinearNetDepth4(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNetDepth4, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def count_num_weights(N):
    count = 1 * HIDDEN_SIZE  # input_layer adds 1
    count += (N - 2) * HIDDEN_SIZE * HIDDEN_SIZE
    count += 1 * HIDDEN_SIZE  # output_layer adds 1

    return count


def calc_min_and_max_eigenval_of_hessian(net, loss_fn, X_tensor, y_tensor, N):
    tensor_train = TensorDataset(X_tensor, y_tensor)
    train_dataloader = DataLoader(tensor_train, batch_size=len(tensor_train))
    n_eigenvalues = count_num_weights(N)
    hessian_comp = hessian(net,
                           loss_fn,
                           dataloader=train_dataloader,
                           cuda=False)

    eigenvalues = hessian_comp.eigenvalues(top_n=n_eigenvalues, maxIter=300, tol=1e-4)
    max_eigenvalue = eigenvalues[0][0]
    min_eigenvalue = eigenvalues[0][-1]

    if max_eigenvalue < min_eigenvalue:
        print("WEIRED!")
        print("min eigenvalue: " + str(min_eigenvalue))
        print("max eigenvalue: " + str(max_eigenvalue))

    return min_eigenvalue, max_eigenvalue


def training_loop(model, N):
    X_tensor, y_tensor, X, y = generate_data()

    # Define the loss function
    loss_function = nn.MSELoss()

    # Define the optimizer
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize lists for storing the loss, gradient magnitude, and eigenvalues
    loss_list = []
    grad_mag_list = []
    eigenvalues_max_list = []
    eigenvalues_min_list = []

    # Training loop
    num_epochs = 100

    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X_tensor)

        # Compute the loss
        loss = loss_function(y_pred, y_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute gradient magnitude
        grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        grad_mag = torch.norm(grad_vec)
        grad_mag_list.append(grad_mag.item())

        # Compute eigenvalues of the Hessian
        min_eigenval, max_eigenval = calc_min_and_max_eigenval_of_hessian(model, loss_function, X_tensor, y_tensor, N)
        eigenvalues_max_list.append(max_eigenval)
        eigenvalues_min_list.append(min_eigenval)

        # Store the loss
        loss_list.append(loss.item())

        # Print the loss every 100 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    predicted_line = model(X_tensor).detach().numpy()

    return predicted_line, loss_list, grad_mag_list, eigenvalues_max_list, eigenvalues_min_list, X, y


if __name__ == '__main__':
    model2 = LinearNetDepth2(1, 1)
    model3 = LinearNetDepth3(1, 1)
    model4 = LinearNetDepth4(1, 1)

    models = [model2, model3, model4]
    model_lines = []
    model_loss = []
    model_grad_mag = []
    model_eigenvalues_max = []
    model_eigenvalues_min = []

    for i, model in enumerate(models, start=2):
        predicted_line, loss_list, grad_mag_list, eigenvalues_max_list, eigenvalues_min_list, X, y = training_loop(
            model, N=i)
        model_lines.append(predicted_line)
        model_loss.append(loss_list)
        model_grad_mag.append(grad_mag_list)
        model_eigenvalues_max.append(eigenvalues_max_list)
        model_eigenvalues_min.append(eigenvalues_min_list)

    # Plot the loss
    plt.figure(figsize=(8, 6))
    for i, loss in enumerate(model_loss, start=2):
        plt.plot(loss, label=f"Model {i}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('Training Loss')

    # Plot the gradient magnitude
    plt.figure(figsize=(8, 6))
    for i, grad_mag in enumerate(model_grad_mag, start=2):
        plt.plot(grad_mag, label=f"Model {i}")
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.title('Magnitude of Model Gradient')
    plt.legend()
    plt.savefig('Magnitude of Model Gradient')

    # Plot the maximal and minimal eigenvalues of the Hessian
    plt.figure(figsize=(8, 6))
    for i, (max_eigenval, min_eigenval) in enumerate(zip(model_eigenvalues_max, model_eigenvalues_min), start=2):
        plt.plot(max_eigenval, label=f"Model {i} (Max)")
        plt.plot(min_eigenval, label=f"Model {i} (Min)")
    plt.xlabel('Epoch')
    plt.ylabel('Eigenvalues')
    plt.title('Maximal and Minimal Eigenvalues of Model Hessian')
    plt.legend()
    plt.savefig('Maximal and Minimal Eigenvalues of Model Hessian')

    # Plot the data and the predicted lines
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, s=10)
    for i, model_line in enumerate(model_lines, start=2):
        plt.plot(X, model_line, label=f"Model {i}")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Synthetic Regression Dataset with Predicted Lines')
    plt.legend()
    plt.savefig('Synthetic Regression Dataset with Predicted Lines')
