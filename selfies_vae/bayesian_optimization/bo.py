import torch
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model


class BayesianOptimizer:
    def __init__(self, train_x, train_y, bounds):
        """
        Initializes the Bayesian Optimizer.

        :param train_x: Initial training samples, shape (N, d) as a Tensor.
        :param train_y: Initial objective values, shape (N, 1) as a Tensor.
        :param bounds: Search space boundaries, shape (2, d) as a Tensor.
        """
        self.train_x = train_x
        self.train_y = train_y
        self.bounds = bounds
        self.gp = None
        self.acq_function = None

    def initialize_gp(self):
        """Initializes the GP model."""
        self.gp = SingleTaskGP(self.train_x, self.train_y)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)

    def update_gp(self, new_x, new_y):
        """Updates the GP model with new sampled points.

        :param new_x: New sampled points, shape (1, d) as a Tensor.
        :param new_y: New objective values, shape (1, 1) as a Tensor.
        """
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])
        self.gp.set_train_data(self.train_x, self.train_y, strict=False)

    def acquire_next_point(self):
        """Finds the next sampling point by optimizing the acquisition function.

        :return: The next sampling point (Tensor) and its acquisition value (Tensor).
        """
        self.acq_function = ExpectedImprovement(self.gp, best_f=self.train_y.max())
        candidate, acq_value = optimize_acqf(
            acq_function=self.acq_function,
            bounds=self.bounds,
            q=1,  # Optimize one sample at a time
            num_restarts=10,
            raw_samples=50,
        )
        return candidate, acq_value

    def optimize(self, objective_function, num_iters=10):
        """Runs the full Bayesian Optimization process.

        :param objective_function: A function that takes samples and returns objective values.
        :param num_iters: Number of optimization iterations.
        :return: A list of sampled points and their corresponding objective values.
        """
        self.initialize_gp()
        samples = []
        for i in range(num_iters):
            print(f"Iteration {i + 1}/{num_iters}")
            candidate, acq_value = self.acquire_next_point()
            new_y = objective_function(candidate)  # Evaluate the objective function
            self.update_gp(candidate, new_y)
            samples.append((candidate, new_y))
        return samples
