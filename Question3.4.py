import functools

import torch
from matplotlib import pyplot as plt
from torch import nn, optim

from base import LinearNetDepth2, LinearNetDepth3, generate_data, calculate_matrix_power


def training_loop(model, X_tensor, y_tensor, num_epochs):
    # Define the loss function
    loss_function = nn.MSELoss()

    # Define the optimizer
    learning_rate = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    N = len(list((model.children())))
    empirical_e2e_loss_list = []
    calculated_e2e_loss_list = []

    # Calculate initial e2e matrix from the results after the step
    calculated_e2e_matrix = functools.reduce(lambda a, b: torch.mm(
        b.clone().detach(), a.clone().detach()
    ), model.parameters()).requires_grad_(True)

    for epoch in range(num_epochs):
        # Calculate consts for update of calculated_e2e_matrix, before the model changes
        WtWtT = torch.mm(calculated_e2e_matrix, calculated_e2e_matrix.T).detach()
        WtTWt = torch.mm(calculated_e2e_matrix.T, calculated_e2e_matrix).detach()

        # Forward pass
        y_empirical = model(X_tensor)
        y_calculated = torch.mm(calculated_e2e_matrix, X_tensor.T).T

        calculated_e2e_matrix.retain_grad()
        # Compute the empirical_loss
        empirical_loss = loss_function(y_empirical, y_tensor)
        calculated_loss = loss_function(y_calculated, y_tensor)

        calculated_loss.backward()
        delta_l = calculated_e2e_matrix.grad

        # Backward pass and optimization
        optimizer.zero_grad()
        empirical_loss.backward()
        optimizer.step()

        empirical_e2e_loss_list.append(empirical_loss.detach())
        calculated_e2e_loss_list.append(calculated_loss.detach())

        # Update W(t+1) from W(t)
        offset = torch.zeros_like(calculated_e2e_matrix)
        for j in range(1, N + 1):
            offset += calculate_matrix_power(WtWtT, (j - 1) / N) * \
                     delta_l * \
                     calculate_matrix_power(WtTWt, (N - j) / N)

        with torch.no_grad():
            calculated_e2e_matrix -= learning_rate * offset
        # Detach matrix so it wouldn't be related to previous calculations anymore
        calculated_e2e_matrix = calculated_e2e_matrix.clone().detach().requires_grad_(True)

        # Print the loss every 100 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Empirical Loss: {empirical_loss.item():.4f}, '
                  f'Calculated Loss: {calculated_loss.item():.4f}')

    print("")
    return empirical_e2e_loss_list, calculated_e2e_loss_list


def main():
    torch.manual_seed(42)
    model2 = LinearNetDepth2(1, 1, is_bias=False)
    model3 = LinearNetDepth3(1, 1, is_bias=False)

    models = [model2, model3]
    model_empirical_e2e_matrix_list = []
    model_calculated_e2e_matrix_list = []

    n_samples = 10000
    input_size = 100
    X_tensor, y_tensor, X, y = generate_data(n_samples=(n_samples, input_size))

    for i, model in enumerate(models, start=2):
        num_epochs = 200
        empirical_e2e_matrix_list, calculated_e2e_matrix_list = training_loop(model, X_tensor, y_tensor, num_epochs)
        model_empirical_e2e_matrix_list.append(empirical_e2e_matrix_list)
        model_calculated_e2e_matrix_list.append(calculated_e2e_matrix_list)

    # Plot the gradient magnitude
    plt.figure(figsize=(8, 6))

    for i, empirical_e2e_matrix_list_ in enumerate(model_empirical_e2e_matrix_list, start=2):
        plt.plot(empirical_e2e_matrix_list_, label=f"Model {i} Layers - empirical")
    for i, calculated_e2e_matrix_list_ in enumerate(model_calculated_e2e_matrix_list, start=2):
        plt.plot(calculated_e2e_matrix_list_, label=f"Model {i} Layers - calculated")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss for each model for W')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
