import os
import yaml
import torch
import numpy as np
import pandas as pd
import selfies as sf
from .utils import multiple_selfies_to_hot
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract SELFIES and SMILES encodings and their corresponding metadata from a dataset.
def get_selfie_and_smiles_encodings_for_dataset(file_path):
    df = pd.read_csv(file_path)
    smiles_list = np.asarray(df.smiles)
    
    # Generate SMILES alphabet and determine the largest SMILES length.
    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # Padding character

    # Convert SMILES to SELFIES and extract SELFIES metadata.
    selfies_list = list(map(sf.encoder, smiles_list))
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')  # Padding token
    selfies_alphabet = list(all_selfies_symbols)
    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    return selfies_list, selfies_alphabet, largest_selfies_len

# Preprocess dataset and save the processed data and metadata to files.
def preprocess_and_save_data(settings_file):
    # Load settings from the YAML file.
    if os.path.exists(settings_file):
        settings = yaml.safe_load(open(settings_file, "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return    
    file_name_smiles = settings['data']['smiles_file']

    # Retrieve encodings and metadata for SELFIES and SMILES.
    encoding_list, encoding_alphabet, largest_molecule_len = \
        get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

    # Generate one-hot encodings for the dataset.
    data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
    
    # Save processed data to a file.
    np.save(settings['data']['encoded_data_file'], data)
    print(f"Saved one-hot encoded data to 'encoded_data.npy'.")
    
    # Split data into train and validation datasets.
    data = torch.tensor(data, dtype=torch.float).to(DEVICE)
    train_valid_test_size = [0.5, 0.5, 0.0]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])
    data_train = data[:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]

    # Save train and validation datasets as .pt files.
    torch.save(data_train, settings['data']['data_train_file'])
    torch.save(data_valid, settings['data']['data_valid_file'])

    # Save the encoding alphabet to a JSON file.
    with open(settings['data']['encoding_alphabet_file'], "w") as f:
        json.dump(encoding_alphabet, f, indent=4)
    print("Saved encoding_alphabet as JSON.")
