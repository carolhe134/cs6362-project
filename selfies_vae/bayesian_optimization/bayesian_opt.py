import torch
from rdkit.Chem import RDConfig, QED
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from bo import BayesianOptimizer
from selfiesvae.train import load_data, initialize_model
from selfiesvae.utils import one_hot_to_selfies, selfies_to_mol_list
from selfiesvae.evaluate import latent_to_one_hot

# Constants
SETTINGS_FILE = "settings_50k_optim.yml"
ENCODER_FILE = "encoder.pt"
DECODER_FILE = "decoder.pt"
NUM_ITERATIONS = 3 # later change to 10
BO_FILE = "bo_progress_50k_optim.yml"
INITIAL_SAMPLE_FACTOR = 1 # later change to 5

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading VAE model")

data, _, _, encoding_alphabet, settings = load_data(SETTINGS_FILE)
len_max_molec = data.shape[1]
len_alphabet = data.shape[2]

vae_encoder, vae_decoder, _, _ = initialize_model(settings, data, encoding_alphabet)
vae_encoder.load_state_dict(torch.load(ENCODER_FILE))
vae_decoder.load_state_dict(torch.load(DECODER_FILE))

vae_encoder.eval()
vae_decoder.eval()

print("VAE model loaded successfully.")

def calculate_molecule_properties(mol_list):
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

def calculate_objective_values(property_list):
    objective_values = []
    for i, props in enumerate(property_list):
        qed = props.get("qed")
        sas = props.get("sas")
        
        if qed is None or sas is None:
            print(f"Skipping element {i} due to missing values (QED={qed}, SAS={sas}).")
            objective_values.append(None)
        else:
            objective_value = 5 * qed - sas
            objective_values.append(objective_value)
    
    return objective_values

def objective_function(latent_points):  
    # 1. Decode latent points (batch_size, latent_dim) to one-hot SELFIES (batch_size, len_max_molec, len_alphabet)
    out_one_hot = latent_to_one_hot(vae_decoder, latent_points, len_max_molec, len_alphabet)

    # 2. Convert one-hot SELFIES (batch_size, len_max_molec, len_alphabet) to SELFIES strings
    selfies_strings = one_hot_to_selfies(out_one_hot, encoding_alphabet)

    # 3. Convert SELFIES to mol
    mol_list = selfies_to_mol_list(selfies_strings)

    # 4. Calculate QED and SAS for each mol
    property_list = calculate_molecule_properties(mol_list)

    # 5. Compute the objective (5 * qed - sas)
    objective_values = calculate_objective_values(property_list)

    return torch.tensor(objective_values)

print("Initializing training data")

latent_dim = settings['decoder']['latent_dimension']
num_initial_samples = INITIAL_SAMPLE_FACTOR * latent_dim
train_x = torch.rand((num_initial_samples, latent_dim))
train_y = objective_function(train_x).unsqueeze(-1)  # Compute the objective values
bounds = torch.tensor([[0.0] * latent_dim, [1.0] * latent_dim])  # Search space bounds

# Initialize the Bayesian Optimizer
optimizer = BayesianOptimizer(train_x, train_y, bounds)

# Run Bayesian Optimization
samples = optimizer.optimize(objective_function, num_iters=NUM_ITERATIONS)

# Print the optimization results
with open(BO_FILE, 'w') as file:
    file.write("Iteration\tCandidate\tValue\n")
    for i, (candidate, value) in enumerate(samples):       
        file.write(f"{i + 1}\t{candidate.cpu().numpy()}\t{value.item()}\n")