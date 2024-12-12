import numpy as np
import selfies as sf
from rdkit.Chem import MolFromSmiles
from rdkit import Chem

# Convert a single SMILES string to one-hot encoding.
def smile_to_hot(smile, largest_smile_len, alphabet):
    char_to_int = {c: i for i, c in enumerate(alphabet)}
    smile += ' ' * (largest_smile_len - len(smile))
    integer_encoded = [char_to_int[char] for char in smile]
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0] * len(alphabet)
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)

# Convert a list of SMILES strings to one-hot encoding.
def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    hot_list = []
    for smile in smiles_list:
        _, onehot_encoded = smile_to_hot(smile, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

# Convert a single SELFIES string to one-hot encoding.
def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    symbol_to_int = {c: i for i, c in enumerate(alphabet)}
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]
    onehot_encoded = []
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)

# Convert a list of SELFIES strings to one-hot encoding.
def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    hot_list = []
    for selfie in selfies_list:
        _, onehot_encoded = selfies_to_hot(selfie, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

# Check if a SMILES string corresponds to a valid molecule.
def is_correct_smiles(smiles):
    if not smiles:
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False

# Convert one hot SELFIES to SELFIES strings
def one_hot_to_selfies(one_hot, alphabet):
    indices = one_hot.argmax(dim=2)
    selfies_strings = []
    for sequence in indices:  # Iterate over batch
        selfies_string = ''.join(
            [alphabet[idx] for idx in sequence if alphabet[idx] != "[nop]"]
        )
        selfies_strings.append(selfies_string)
    return selfies_strings

# Convert from SELFIES strings to molecules
def selfies_to_mol_list(selfies_strings):
    mol_list = []
    for selfies in selfies_strings:
        try:
            # Convert SELFIES to SMILES
            smiles = sf.decoder(selfies)
            if smiles:
                # Convert SMILES to RDKit Mol object
                mol = Chem.MolFromSmiles(smiles)
                mol_list.append(mol)  # Append the Mol object (or None if invalid)
            else:
                mol_list.append(None)  # Append None for invalid SMILES
        except Exception as e:
            # Handle any unexpected errors
            print(f"Error processing SELFIES '{selfies}': {e}")
            mol_list.append(None)
    return mol_list
