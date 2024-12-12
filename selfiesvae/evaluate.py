import torch
from torch import nn
import selfies as sf
import numpy as np
from utils import is_correct_smiles

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute the Evidence Lower Bound (ELBO) for VAE.
def elbo(x, x_hat, mus, log_vars, KLD_alpha):
    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(dim=1)

    criterion = nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1.0 + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld, recon_loss, kld

# Assess the reconstruction quality of input and output tensors.
def recon_quality(x, x_hat):
    x_indices = x.reshape(-1, x.shape[2]).argmax(dim=1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(dim=1)

    differences = 1.0 - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0.0, max=1.0).double()
    quality = 100.0 * torch.mean(differences)

    return quality.detach().cpu().numpy()

# Compute reconstruction quality and validation loss for the validation dataset.
def quality_in_valid_set(vae_encoder, vae_decoder, data_valid, batch_size, KLD_alpha):
    data_valid = data_valid[torch.randperm(data_valid.size(0))]  # Shuffle data
    num_batches_valid = len(data_valid) // batch_size

    quality_list = []
    valid_loss = 0.0

    for batch_idx in range(min(25, num_batches_valid)):
        batch = data_valid[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        _, trg_len, _ = batch.size()

        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)
        out_one_hot = torch.zeros_like(batch, device=DEVICE)

        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        # Assess reconstruction quality
        quality = recon_quality(batch, out_one_hot)
        quality_list.append(quality)

        # Compute validation loss
        loss, _, _ = elbo(batch, out_one_hot, mus, log_vars, KLD_alpha)
        valid_loss += loss.item()

    # Average validation loss
    valid_loss /= num_batches_valid

    return np.mean(quality_list).item(), valid_loss

# Sample the latent space and generate a sequence using the decoder.
def sample_latent_space(vae_encoder, vae_decoder, sample_len):
    vae_encoder.eval()
    vae_decoder.eval()

    gathered_atoms = []
    latent_point = torch.randn(1, 1, vae_encoder.latent_dimension, device=DEVICE)
    hidden = vae_decoder.init_hidden()

    softmax = nn.Softmax(dim=0)

    for _ in range(sample_len):
        out_one_hot, hidden = vae_decoder(latent_point, hidden)
        out_one_hot = softmax(out_one_hot.flatten().detach())
        gathered_atoms.append(out_one_hot.argmax(dim=0).item())

    vae_encoder.train()
    vae_decoder.train()

    return gathered_atoms

# Assess the quality of molecules sampled from the latent space.
def latent_space_quality(vae_encoder, vae_decoder, alphabet, sample_num, sample_len):
    total_correct = 0
    all_correct_molecules = set()

    for _ in range(sample_num):
        molecule_str = ''.join(alphabet[i] for i in sample_latent_space(vae_encoder, vae_decoder, sample_len))
        molecule = sf.decoder(molecule_str.replace(' ', ''))

        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)

    return total_correct, len(all_correct_molecules)

# Convert latent vectors to one hot SELFIES
def latent_to_one_hot(vae_decoder, latent_points, len_max_molec, len_alphabet):
    batch_size = latent_points.shape[0]
    latent_points = latent_points.unsqueeze(0)
    hidden = vae_decoder.init_hidden(batch_size=batch_size)
    out_one_hot = torch.zeros(batch_size, len_max_molec, len_alphabet, device=DEVICE)
    for seq_idx in range(len_max_molec):
        out_line, hidden = vae_decoder(latent_points, hidden)
        out_one_hot[:, seq_idx, :] = out_line[0]
    return out_one_hot