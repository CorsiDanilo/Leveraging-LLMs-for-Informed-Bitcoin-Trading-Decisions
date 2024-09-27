from torch import nn
import torch
import numpy as np
from models.utils.models_utils import process_embeddings

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    """ 
    Positional encoding for Transformer model

    Parameters:
        model_dim (int): Model dimension
        dropout (float): Dropout rate
        max_len (int): Maximum length of sequence

    Returns:
        out (torch.Tensor): Output tensor
    """
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

# Model definition using Transformer
class TransformerBase(nn.Module):
    """ 
    Transformer model with embeddings for categorical features and continuous features

    Parameters:
        embedding_layers (list): List of embedding layers for categorical features
        num_continuous_features (int): Number of continuous features
        embedding_dim (int): Dimension of embeddings
        model_dim (int): Model dimension
        num_heads (int): Number of heads in multi-head attention
        num_layers (int): Number of Transformer layers
        output_size (int): Output size
        dropout (float): Dropout rate

    Returns:
        out (torch.Tensor): Output tensor
    """
    def __init__(self, parameters, dropout=0.1):
        super(TransformerBase, self).__init__()
        self.embedding_layers = parameters['embedding_layers']
        self.embedding_dim = parameters['embedding_dim']
        self.num_continuous_features = parameters['num_continuous_features']
        self.model_dim = parameters['model_dim']
        self.num_heads = parameters['num_heads']
        self.num_layers = parameters['num_layers']
        self.output_size = parameters['output_size']
        self.dropout = parameters['dropout']

        # Linear layer to project input features to model dimension
        self.encoder = nn.Linear(self.num_continuous_features + self.embedding_dim, self.model_dim)
        
        # Linear layer to project input features to model dimension
        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
        
        # Linear layer to project input features to model dimension
        self.encoder_layers = nn.TransformerEncoderLayer(self.model_dim, self.num_heads, dim_feedforward=self.model_dim * 4, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.num_layers)
        
        # Final linear layer to project to output size
        self.decoder = nn.Linear(self.model_dim, self.output_size)

    def forward(self, x_continuous, x_categorical):
        # Process embeddings for categorical features
        x_combined = process_embeddings(self.embedding_layers, x_categorical, x_continuous)

        # Project to model dimension
        out = self.encoder(x_combined)

        # Apply positional encoding
        out = self.pos_encoder(out)

        # Apply Transformer encoder
        out = self.transformer_encoder(out)

        # Project to output size
        out = self.decoder(out[:, -1, :])

        return out