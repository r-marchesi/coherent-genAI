import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, n_layers, shrink_exponent= 1.0):
        """
        Args:
            input_dim (int): Number of features in the input data.
            bottleneck_dim (int): Dimensionality of the latent representation.
            n_layers (int): Number of layers in the encoder (and decoder).
            shrink_exponent (float): Exponent controlling progression of layer sizes; values >1 make the initial reductions steeper.
        """
        super(FlexibleAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.shrink_exponent = shrink_exponent

        # Compute encoder layer sizes via geometric progression
        # For exp >1, early layers shrink more aggressively
        self.encoder_sizes = []
        for i in range(1, n_layers + 1):
            # base fraction
            base_frac = i / float(n_layers)
            # adjusted fraction inverts exponent to steepen initial drop
            frac = base_frac ** (1.0 / self.shrink_exponent)
            size = int(input_dim * (bottleneck_dim / input_dim) ** frac)
            size = max(size, bottleneck_dim)
            self.encoder_sizes.append(size)

        # Build encoder network
        encoder_modules = []
        prev_dim = input_dim
        for h_dim in self.encoder_sizes:
            encoder_modules += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True)
            ]
            prev_dim = h_dim
        # Bottleneck layer
        encoder_modules.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_modules)

        # Build decoder (mirror of encoder)
        decoder_modules = []
        decoder_sizes = list(reversed(self.encoder_sizes))
        prev_dim = bottleneck_dim
        for h_dim in decoder_sizes:
            decoder_modules += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True)
            ]
            prev_dim = h_dim
        # Output layer
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z)

    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor):
        return F.mse_loss(recon_x, x, reduction='mean')
