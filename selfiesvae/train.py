import os
import torch
from torch import nn
import numpy as np
import yaml
import json

from log import *
from evaluate import *
from models import *

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data and settings from a YAML file
def load_data(settings_file):
    if not os.path.exists(settings_file):
        raise FileNotFoundError(f"Settings file '{settings_file}' not found.")

    settings = yaml.safe_load(open(settings_file, "r"))

    if not os.path.exists(settings['encoded_data_file']):
        raise FileNotFoundError("Encoded data file not found. Run preprocessing first.")

    data = np.load(settings['encoded_data_file'])
    data_train = torch.load(settings['data_train_file'])
    data_valid = torch.load(settings['data_valid_file'])

    with open(settings['encoding_alphabet_file'], "r") as f:
        encoding_alphabet = json.load(f)

    return data, data_train, data_valid, encoding_alphabet, settings

# Initialize VAE model components
def initialize_model(settings, data, encoding_alphabet):
    encoder_params = settings['encoder']
    decoder_params = settings['decoder']

    vae_encoder = VAEEncoder(
        in_dimension=data.shape[1] * data.shape[2], **encoder_params
    ).to(DEVICE)

    vae_decoder = VAEDecoder(
        **decoder_params, out_dimension=len(encoding_alphabet)
    ).to(DEVICE)

    print(f"Model initialized on {DEVICE}.")

    return vae_encoder, vae_decoder, settings['data'], settings['training']

# Train the Variational Auto-Encoder (VAE) model
def train_model(
    vae_encoder, vae_decoder,
    data_train, data_valid, num_epochs, batch_size,
    lr_enc, lr_dec, KLD_alpha,
    sample_num, sample_len, alphabet, results_file,
    log_data=False, log_dir=None, model_dir=None
):
    with open(results_file, 'w') as f:
        f.write("")

    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    if log_data:
        log_files = initialize_log_files(log_dir)

    data_train = data_train.clone().detach().to(DEVICE)
    num_batches_train = len(data_train) // batch_size
    quality_valid_list = []
    total_loss = 0

    for epoch in range(num_epochs):
        data_train = data_train[torch.randperm(data_train.size(0))]
        epoch_metrics = {'loss': 0, 'kl_divergence': 0, 'reconstruction_loss': 0}

        for batch_idx in range(num_batches_train):
            start_idx = batch_idx * batch_size
            stop_idx = start_idx + batch_size
            batch = data_train[start_idx:stop_idx]

            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            latent_points = latent_points.unsqueeze(0)
            hidden = vae_decoder.init_hidden(batch_size=batch_size)
            out_one_hot = torch.zeros_like(batch, device=DEVICE)

            for seq_idx in range(batch.shape[1]):
                out_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_idx, :] = out_line[0]

            loss, recon_loss, kld = elbo(batch, out_one_hot, mus, log_vars, KLD_alpha)
            total_loss += loss.item()

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            epoch_metrics['loss'] += loss.item()
            epoch_metrics['kl_divergence'] += kld.item()
            epoch_metrics['reconstruction_loss'] += recon_loss.item()

            if batch_idx % 30 == 0:
                quality_train = recon_quality(batch, out_one_hot)
                quality_valid, valid_loss = quality_in_valid_set(
                    vae_encoder, vae_decoder, data_valid, batch_size, KLD_alpha
                )
                print(f"Epoch {epoch} Batch {batch_idx}: "
                      f"Loss={loss.item():.4f}, Val Loss={valid_loss:.4f}, "
                      f"Train Quality={quality_train:.4f}, Val Quality={quality_valid:.4f}")

        epoch_metrics = {k: v / num_batches_train for k, v in epoch_metrics.items()}
        quality_valid, epoch_valid_loss = quality_in_valid_set(
            vae_encoder, vae_decoder, data_valid, batch_size, KLD_alpha
        )
        quality_valid_list.append(quality_valid)

        # Check reconstruction quality improvements
        quality_increase = len(quality_valid_list) - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.0:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder, alphabet, sample_num, sample_len)
        else:
            corr, unique = -1.0, -1.0

        report = (
            f"Validity: {corr * 100.0 / sample_num:.5f} % | "
            f"Diversity: {unique * 100.0 / sample_num:.5f} % | "
            f"Reconstruction: {quality_valid:.5f} %"
        )
        print(report)

        with open(results_file, 'a') as content:
            content.write(report + '\n')

        if log_data:
            log_metrics(log_files, epoch, {
                **epoch_metrics,
                'validation_loss': epoch_valid_loss
            })
            save_models(vae_encoder, vae_decoder, epoch, model_dir)

        if quality_valid < 70.0 and epoch > 200:
            break

        if quality_increase > 20:
            print("Early stopping criteria met.")
            break

    avg_loss = total_loss / (num_epochs * len(data_train))
    return avg_loss

# Main training function
def train(settings_file, log_data=False, log_dir=None, model_dir=None):
    data, data_train, data_valid, encoding_alphabet, settings = load_data(settings_file)

    vae_encoder, vae_decoder, data_params, training_params = initialize_model(settings, 
                                                                              data, 
                                                                              encoding_alphabet)

    train_model(
        **training_params,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        batch_size=data_params['batch_size'],
        data_train=data_train,
        data_valid=data_valid,
        alphabet=encoding_alphabet,
        sample_len=data.shape[1],
        results_file=data_params['results_name'],
        log_data=log_data,
        log_dir=log_dir,
        model_dir=model_dir
    )