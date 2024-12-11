import torch
from torch import nn

class VAEEncoder(nn.Module):
    # Encoder module for a Variational Autoencoder (VAE).
    # Encodes the input into a latent space representation.
    def __init__(self, input_dim, layer1_dim, layer2_dim, layer3_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Fully connected layers to reduce dimensionality
        self.encoder_nn = nn.Sequential(
            nn.Linear(input_dim, layer1_dim),
            nn.ReLU(),
            nn.Linear(layer1_dim, layer2_dim),
            nn.ReLU(),
            nn.Linear(layer2_dim, layer3_dim),
            nn.ReLU()
        )

        # Layers for computing latent space parameters
        self.mu_layer = nn.Linear(layer3_dim, latent_dim)
        self.log_var_layer = nn.Linear(layer3_dim, latent_dim)

    @staticmethod
    # Reparameterization trick to sample from N(mu, sigma^2) using N(0,1).
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    # Forward pass through the encoder.
    def forward(self, x):
        hidden = self.encoder_nn(x)
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VAEDecoder(nn.Module):
    # Decoder module for a Variational Autoencoder (VAE).
    # Decodes the latent space representation back into the original space.
    def __init__(self, latent_dim, gru_layers, gru_units, output_dim):
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.gru_layers = gru_layers
        self.gru_units = gru_units

        # GRU for sequence decoding
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=gru_units,
            num_layers=gru_layers,
            batch_first=False
        )

        # Fully connected layer for output decoding
        self.output_layer = nn.Linear(gru_units, output_dim)

    # Initialize the hidden state for the GRU.
    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_layers, batch_size, self.gru_units)

    # Forward pass through the decoder.
    def forward(self, z, hidden):
        gru_output, hidden = self.gru(z, hidden)
        decoded = self.output_layer(gru_output)
        return decoded, hidden
