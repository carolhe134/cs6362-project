import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import sys
import os
import numpy as np
import importlib

from rdkit.Chem import RDConfig, QED
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# Path to selfiesvae
selfiesvae_path = os.path.abspath(os.path.join(os.getcwd(), "../../selfiesvae"))
module_name = "selfiesvae"

# Load the module dynamically
spec = importlib.util.spec_from_file_location(module_name, os.path.join(selfiesvae_path, "__init__.py"))
selfiesvae = importlib.util.module_from_spec(spec)
sys.modules[module_name] = selfiesvae
spec.loader.exec_module(selfiesvae)
from selfiesvae.utils import one_hot_to_selfies, selfies_to_mol_list
from selfiesvae.evaluate import latent_to_one_hot

class BayesianOptimizer:
    def __init__(self, latent_dim, bounds, vae_decoder, len_max_mol, len_alphabet, encoding_alphabet, device="cpu"):
        """
        Initialize the Bayesian Optimizer.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            bounds (torch.Tensor): Tensor of shape (2, latent_dim) defining the search bounds.
            device (str): Device to run the computations on.
        """
        self.latent_dim = latent_dim
        self.bounds = bounds
        self.vae_decoder = vae_decoder
        self.len_max_mol = len_max_mol
        self.len_alphabet = len_alphabet
        self.encoding_alphabet = encoding_alphabet
        self.device = device

    def initialize_data(self, initial_points):
        """
        Initialize data for optimization.

        Args:
            initial_points (torch.Tensor): Initial latent points of shape (n_points, latent_dim).
        """
        self.train_x = initial_points.to(self.device)
        self.train_y = self.evaluate_property(self.train_x)

    def _calculate_molecule_properties(self, mol_list):
        results = []
        for i, mol in enumerate(mol_list):
            if mol is None:
                print(f"Molecule {i} is invalid or could not be processed.")
                results.append({"qed": None, "sas": None})
                continue
            
            try:
                # Calculate QED and SAS
                qed = QED.qed(mol)
                sas = sascorer.calculateScore(mol)
                results.append({"qed": qed, "sas": sas})
            except Exception as e:
                print(f"Error calculating properties for molecule {i}: {e}")
                results.append({"qed": None, "sas": None})
        
        return results
    
    def _calculate_objective_values(self, property_list):
        objective_values = []
        for i, props in enumerate(property_list):
            qed = props.get("qed")
            sas = props.get("sas")
            print("QED\n", qed)
            print("SAS\n", sas)
            
            if qed is None or sas is None:
                print(f"Skipping element {i} due to missing values (QED={qed}, SAS={sas}).")
                objective_values.append([None])
            else:
                objective_value = 5 * qed - sas
                objective_values.append([objective_value])
        
        return objective_values

    def evaluate_property(self, latent_points):
        """
        Evaluate the property function for given latent points.

        Args:
            latent_points (torch.Tensor): Latent points to evaluate.

        Returns:
            torch.Tensor: Property values of shape (n_points, 1).
        """
        print("step 0")

        # 1. Decode latent points (batch_size, latent_dim) to one-hot SELFIES (batch_size, len_max_molec, len_alphabet)
        out_one_hot = latent_to_one_hot(self.vae_decoder, latent_points, self.len_max_mol, self.len_alphabet)

        print("step 1")

        # 2. Convert one-hot SELFIES (batch_size, len_max_molec, len_alphabet) to SELFIES strings
        selfies_strings = one_hot_to_selfies(out_one_hot, self.encoding_alphabet)

        print("step 2")

        # 3. Convert SELFIES to mol
        mol_list = selfies_to_mol_list(selfies_strings)

        print("step 3")

        # 4. Calculate QED and SAS for each mol
        property_list = self._calculate_molecule_properties(mol_list)

        print("step 4")

        # 5. Compute the objective (5 * qed - sas)
        objective_values = self._calculate_objective_values(property_list)

        print("objective shape before:", np.array(objective_values).shape)

        print("step 5")

        # objective_values_tensor = torch.tensor(objective_values).reshape(-1, 1)
        objective_values_tensor = torch.tensor(objective_values)
        print("objective shape after:", objective_values_tensor.shape)
        
        return objective_values_tensor

    def train_model(self):
        """
        Train a Gaussian Process model on the current dataset.
        """
        self.gp = SingleTaskGP(self.train_x, self.train_y).to(self.device)
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(self.mll)

    def optimize_acquisition(self):
        """
        Optimize the acquisition function to propose the next point.

        Returns:
            torch.Tensor: Proposed latent point of shape (1, latent_dim).
        """
        acq_func = ExpectedImprovement(model=self.gp, best_f=self.train_y.max())
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        return candidate

    def step(self):
        """
        Perform one step of Bayesian Optimization.

        Returns:
            torch.Tensor: Newly evaluated latent point.
        """
        self.train_model()
        next_point = self.optimize_acquisition()
        next_value = self.evaluate_property(next_point)

        # Update dataset
        self.train_x = torch.cat([self.train_x, next_point])
        self.train_y = torch.cat([self.train_y, next_value])

        return next_point

    def optimize(self, n_steps):
        """
        Run Bayesian Optimization for a specified number of steps.

        Args:
            n_steps (int): Number of optimization steps.

        Returns:
            torch.Tensor: Final dataset of latent points and their properties.
        """
        for i in range(n_steps):
            print(f"Iteration {i + 1}/{n_steps} begin")
            self.step()
            print(f"Iteration {i + 1}/{n_steps} end")
        return self.train_x, self.train_y
