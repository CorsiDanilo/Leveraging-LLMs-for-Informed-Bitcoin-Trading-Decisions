import torch
from torch import nn
import numpy as np
from torch.distributions.normal import Normal
from models.utils.models_utils import process_embeddings

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Model definition using Transformer for Probabilistic Time Series Forecasting
class ProbabilisticTransformer(nn.Module):
    def __init__(self, parameters):
        super(ProbabilisticTransformer, self).__init__()
        self.embedding_layers = parameters['embedding_layers']
        self.embedding_dim = parameters['embedding_dim']
        self.num_continuous_features = parameters['num_continuous_features']
        self.model_dim = parameters['model_dim']
        self.num_heads = parameters['num_heads']
        self.num_layers = parameters['num_layers']
        self.output_size = parameters['output_size']
        self.dropout = parameters['dropout']

        self.encoder = nn.Linear(self.num_continuous_features + self.embedding_dim, self.model_dim)
        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.num_layers)
        self.decoder = nn.Linear(self.model_dim, 2 * self.output_size) # Output mean and std

    def forward(self, x_continuous, x_categorical):
        # Process embeddings for categorical features
        x_combined = process_embeddings(self.embedding_layers, x_categorical, x_continuous)

        out = self.encoder(x_combined)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)
        out = self.decoder(out[:, -1, :])

        # Split the output into mean and std for Gaussian distribution
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.nn.functional.softplus(std)  # Ensure std is positive
        return mean, std

    def loss_fn(self, mean, std, target):
        # Define the negative log likelihood loss for Gaussian distribution
        dist = Normal(mean, std)
        loss = -dist.log_prob(target).mean()
        return loss