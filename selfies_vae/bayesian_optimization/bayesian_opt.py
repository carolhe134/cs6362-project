import torch
import sys
import os
import importlib
from bo import BayesianOptimizer

# Path to selfiesvae
selfiesvae_path = os.path.abspath(os.path.join(os.getcwd(), "../../selfiesvae"))
module_name = "selfiesvae"

# Load the module dynamically
spec = importlib.util.spec_from_file_location(module_name, os.path.join(selfiesvae_path, "__init__.py"))
selfiesvae = importlib.util.module_from_spec(spec)
sys.modules[module_name] = selfiesvae
spec.loader.exec_module(selfiesvae)
from selfiesvae.train import load_data, initialize_model

# Constants
SETTINGS_FILE = "settings_50k_optim.yml"
DECODER_FILE = "decoder.pt"
NUM_ITERATIONS = 30 # later change to 10
BO_FILE = "bo_progress_50k_optim_30iters_5factor_-3to3bounds.yml"
INITIAL_SAMPLE_FACTOR = 5 # later change to 5
LOWER_BOUND = -3.0
UPPER_BOUND = 3.0

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading VAE model")

data, _, _, encoding_alphabet, settings = load_data(SETTINGS_FILE)
len_max_molec = data.shape[1]
len_alphabet = data.shape[2]

_, vae_decoder, _, _ = initialize_model(settings, data, encoding_alphabet)
vae_decoder.load_state_dict(torch.load(DECODER_FILE, map_location=torch.device('cpu')))
vae_decoder.eval()

print("VAE model loaded successfully.")

# Initialize optimizer
latent_dim = settings['decoder']['latent_dimension']
bounds = torch.tensor([[LOWER_BOUND] * latent_dim, [UPPER_BOUND] * latent_dim])
optimizer = BayesianOptimizer(latent_dim, bounds, vae_decoder, len_max_molec, len_alphabet, encoding_alphabet, device=DEVICE)

# Generate initial data
initial_points = torch.rand(INITIAL_SAMPLE_FACTOR * latent_dim, latent_dim)
optimizer.initialize_data(initial_points)

# Run optimization
final_latents, final_properties = optimizer.optimize(NUM_ITERATIONS)

# Save the optimization results
with open(BO_FILE, 'w') as file:
    file.write("Iteration\tLatentVector\tPropertyValue\n")
    for i, (latent, value) in enumerate(zip(final_latents, final_properties)):
        latent_vector_str = ', '.join(map(str, latent.cpu().numpy()))  # Convert latent vector to string
        file.write(f"{i + 1}\t[{latent_vector_str}]\t{value.item()}\n")