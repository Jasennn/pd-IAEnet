"""
A toy script for learning the map from $\exp(-x^2 - y^2)$ to $\sin(2\pi x) \sin(2\pi y)$
on the domain $[-1, 1]^2$ using the PDIAE-Net model. 
The model is trained on three different different discretizations: 32 x 32, 64 x 64, and 128 x 128.
"""

from pdiaenet import PDIAE_Net
import torch
import torch.nn as nn

# Define input and target functions
def input_function(grid):
    x, y = grid[0], grid[1]
    return torch.exp(-x**2 - y**2)

def target_function(grid):
    x, y = grid[0], grid[1]
    return torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)

# Generate grid for given discretization
def generate_grid(discretization: int):
    x = torch.linspace(-1, 1, discretization)
    y = torch.linspace(-1, 1, discretization)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=0)  # (2, discretization, discretization)
    return grid

# Initialize the model
model = PDIAE_Net(in_channels=3, out_channels=1, width=64, k_rank=2, modes=12, num_blocks=4)

# Training parameters
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
batch_size = 1
discretizations = [32, 64, 128]

# Training loop with separate batches for each discretization
for epoch in range(epochs):
    # Randomly shuffle and process discretizations in this epoch
    for discretization in discretizations:
        # Generate a batch for the current discretization
        batch_inputs = []
        batch_targets = []
        
        for _ in range(batch_size):
            grid = generate_grid(discretization)  # (2, discretization, discretization)
            input_func = input_function(grid)
            target = target_function(grid)  # (discretization, discretization)

            combined_input = torch.cat([grid, input_func.unsqueeze(0)], dim=0) # (3, discretization, discretization)
            
            batch_inputs.append(combined_input)  # (2, discretization, discretization)
            batch_targets.append(target.unsqueeze(0))  # (1, discretization, discretization)

        # Stack inputs and targets to form the batch
        batch_inputs = torch.stack(batch_inputs, dim=0)  # (batch_size, 2, discretization, discretization)
        batch_targets = torch.stack(batch_targets, dim=0)  # (batch_size, 1, discretization, discretization)

        # Training step
        optimizer.zero_grad()
        output = model(batch_inputs)  # (batch_size, 1, discretization, discretization)

        loss = criterion(output, batch_targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

