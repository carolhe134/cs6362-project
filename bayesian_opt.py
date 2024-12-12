import torch
from src.bo import BayesianOptimizer

def objective_function(x):
    qed = x[:, 0]  # Need to be replaced
    sas = x[:, 1]  # Need to be replaced
    return 5 * qed - sas

# Initialize training data
train_x = # Need to add
train_y = objective_function(train_x).unsqueeze(-1)  # Compute the objective values
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # Search space bounds (assume [0, 1] for simplicity)

# Initialize the Bayesian Optimizer
optimizer = BayesianOptimizer(train_x, train_y, bounds)

# Run Bayesian Optimization
samples = optimizer.optimize(objective_function, num_iters=20)

# Print the optimization results
for i, (candidate, value) in enumerate(samples):
    print(f"Iteration {i + 1}: Candidate={candidate.numpy()}, Value={value.item()}")
