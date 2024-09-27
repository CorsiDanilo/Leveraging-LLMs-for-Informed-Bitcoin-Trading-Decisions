from torch import nn
import torch
import numpy as np
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

# Auto-Correlation Mechanism: captures the relationships between different positions in the input sequence by computing attention weights based on the similarity between queries and keys. This allows the model to focus on relevant parts of the input sequence when making predictions.
class AutoCorrelation(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.num_heads = num_heads  # Number of attention heads
        self.model_dim = model_dim  # Dimension of the model
        self.head_dim = model_dim // num_heads  # The dimension of each attention head, calculated by dividing the model dimension by the number of heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for the queries, computed as the inverse square root of the head dimension to prevent large dot products

        # Linear layer that projects the input tensor into three separate tensors for queries (Q), keys (K), and values (V)
        # Model dimension is multiplied by 3 to create three separate tensors
        self.qkv_proj = nn.Linear(model_dim, model_dim * 3)
        # Linear layer that projects the concatenated output of all attention heads back to the original model dimension
        self.out_proj = nn.Linear(model_dim, model_dim)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()  # Get the dimensions of the input tensor

        # The input tensor x is projected into a single tensor qkv containing queries, keys, and values using the qkv_proj layer
        qkv = self.qkv_proj(x)
        # The qkv tensor is reshaped to separate the heads and then split into three tensors: q, k, and v
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1).chunk(3, dim=2)  # (num_heads, batch_size, 3 * head_dim, seq_len)
        q, k, v = qkv  # Split the tensor into queries, keys, and values

        q = q * self.scale  # The queries q are scaled by the factor self.scale to ensure numerical stability
        # Compute the auto-correlation (similarity between queries and keys)
        # The auto-correlation is computed as the dot product between the queries and keys using torch.einsum
        auto_correlation = torch.einsum('bhld,bhmd->bhlm', q, k)
        # Apply softmax to get the attention weights
        auto_weights = torch.softmax(auto_correlation, dim=-1)
        # Apply dropout to the attention weights
        auto_weights = self.dropout(auto_weights)

        # Apply the attention weights to the values
        # The attention weights are obtained by applying the softmax function to the auto-correlation tensor along the last dimension
        out = torch.einsum('bhlm,bhmd->bhld', auto_weights, v)
        # Reshape the output to combine the heads
        out = out.contiguous().view(batch_size, seq_len, self.model_dim)
        # Project the output back to the original model dimension
        out = self.out_proj(out)
        return out

# Autoformer Encoder Layer
class AutoformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(AutoformerEncoderLayer, self).__init__()
        # Initialize the AutoCorrelation layer
        self.auto_correlation = AutoCorrelation(model_dim, num_heads, dropout)
        
        # Feed-forward network: First linear layer expands the dimensionality by a factor of 4
        # Expand and compress the dimensionality to model_dim * 4 for:
        # 1. Increasing the capacity of the model
        # 2. Allowing the model to learn more complex patterns
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        # Feed-forward network: Second linear layer reduces the dimensionality back to original
        self.linear2 = nn.Linear(model_dim * 4, model_dim)
        
        # Layer normalization for stabilizing the training and helping with gradient flow
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        # Dropout layers for the residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Apply the first layer normalization
        x2 = self.norm1(x)
        # Pass the normalized input through the AutoCorrelation layer and apply dropout
        # Add the result to the original input (residual connection)
        x = x + self.dropout1(self.auto_correlation(x2))
        
        # Apply the second layer normalization
        x2 = self.norm2(x)
        # Pass the normalized input through the first linear layer, apply dropout,
        # then pass through the second linear layer, and finally apply dropout again
        # Add the result to the original input (residual connection)
        x = x + self.dropout2(self.linear2(self.dropout(self.linear1(x2))))
        
        # Return the output of the encoder layer
        return x

class Autoformer(nn.Module):
    def __init__(self, parameters):
        super(Autoformer, self).__init__()
        
        # Initialize model parameters from the dictionary 'parameters'
        self.embedding_layers = parameters['embedding_layers'] # List of embedding layers for categorical features
        self.embedding_dim = parameters['embedding_dim'] # Dimension of the embeddings for categorical features
        self.num_continuous_features = parameters['num_continuous_features'] # Number of continuous features
        self.model_dim = parameters['model_dim'] # Dimension of the model (and of the encoder layers)
        self.num_heads = parameters['num_heads'] # Number of attention heads in the AutoCorrelation mechanism
        self.num_layers = parameters['num_layers'] # Number of encoder layers in the model
        self.output_size = parameters['output_size'] # Size of the output
        self.dropout = parameters['dropout'] # Dropout rate to apply during training

        # Linear layer to combine continuous and categorical features into model_dim
        self.encoder = nn.Linear(self.num_continuous_features + self.embedding_dim, self.model_dim)
        
        # Positional encoding to add temporal information to the input embeddings
        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
        
        # Stack of Autoformer encoder layers
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(self.model_dim, self.num_heads, self.dropout) for _ in range(self.num_layers)
        ])
        
        # Linear layer to transform the final output to the desired output size
        self.decoder = nn.Linear(self.model_dim, self.output_size)

    def forward(self, x_continuous, x_categorical):
        # Process embeddings for categorical features
        # 'process_embeddings' is a utility function that combines embeddings with continuous features
        x_combined = process_embeddings(self.embedding_layers, x_categorical, x_continuous)

        # Apply the initial linear layer to get the input ready for the encoder layers
        out = self.encoder(x_combined)
        
        # Add positional encodings to the input
        out = self.pos_encoder(out)
        
        # Pass the input through each of the encoder layers sequentially
        for layer in self.encoder_layers:
            out = layer(out)
        
        # Apply the final linear layer to get the output of the desired size
        # Here we are taking the output of the last time step (i.e., out[:, -1, :])
        out = self.decoder(out[:, -1, :])
        
        return out

# import torch
# from torch import nn
# import numpy as np
# from models.utils.models_utils import process_embeddings

# # Positional Encoding for Transformer
# class PositionalEncoding(nn.Module):
#     def __init__(self, model_dim, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         # Compute the positional encodings once in log space
#         pe = torch.zeros(max_len, model_dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# # Decomposition block for time-series
# class DecompositionBlock(nn.Module):
#     def __init__(self, kernel_size):
#         super(DecompositionBlock, self).__init__()
#         self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)

#     def forward(self, x):
#         # Trend component
#         trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
#         # Seasonal component
#         seasonal = x - trend
#         return trend, seasonal

# # Auto-Correlation mechanism for time-series
# class AutoCorrelation(nn.Module):
#     def __init__(self, model_dim, num_heads):
#         super(AutoCorrelation, self).__init__()
#         self.num_heads = num_heads
#         self.model_dim = model_dim
#         self.query_projection = nn.Linear(model_dim, model_dim)
#         self.key_projection = nn.Linear(model_dim, model_dim)
#         self.value_projection = nn.Linear(model_dim, model_dim)
#         self.output_projection = nn.Linear(model_dim, model_dim)
    
#     def forward(self, x):
#         query = self.query_projection(x)
#         key = self.key_projection(x)
#         value = self.value_projection(x)
        
#         # Calculate Auto-Correlation
#         auto_corr = torch.einsum('bqd,bkd->bqk', query, key)
#         auto_corr = torch.softmax(auto_corr / np.sqrt(self.model_dim), dim=-1)
        
#         out = torch.einsum('bqk,bkd->bqd', auto_corr, value)
#         out = self.output_projection(out)
#         return out

# # Model definition using Autoformer
# class Autoformer(nn.Module):
#     def __init__(self, parameters):
#         super(Autoformer, self).__init__()
#         self.embedding_layers = parameters['embedding_layers']
#         self.embedding_dim = parameters['embedding_dim']
#         self.num_continuous_features = parameters['num_continuous_features']
#         self.model_dim = parameters['model_dim']
#         self.num_heads = parameters['num_heads']
#         self.num_layers = parameters['num_layers']
#         self.output_size = parameters['output_size']
#         self.dropout = parameters['dropout']
#         self.kernel_size = 7 # Adjust this based on your data's periodicity

#         # Linear layer to project input features to model dimension
#         self.encoder = nn.Linear(self.num_continuous_features + self.embedding_dim, self.model_dim)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
        
#         # Decomposition block
#         self.decomposition = DecompositionBlock(kernel_size=self.kernel_size)
        
#         # Auto-Correlation mechanism
#         self.auto_correlation = AutoCorrelation(self.model_dim, self.num_heads)
        
#         # Encoder layers
#         self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(self.model_dim, self.num_heads, dim_feedforward=self.model_dim * 4, dropout=self.dropout) for _ in range(self.num_layers)])
        
#         # Final linear layer to project to output size
#         self.decoder = nn.Linear(self.model_dim, self.output_size)

#     def forward(self, x_continuous, x_categorical):
#         # Process embeddings for categorical features
#         x_combined = process_embeddings(self.embedding_layers, x_categorical, x_continuous)

#         # Project to model dimension
#         out = self.encoder(x_combined)

#         # Apply positional encoding
#         out = self.pos_encoder(out)

#         # Decompose input
#         trend, seasonal = self.decomposition(out)

#         # Apply Auto-Correlation mechanism
#         out = self.auto_correlation(seasonal)

#         # Apply Transformer encoder layers
#         for layer in self.encoder_layers:
#             out = layer(out)

#         # Combine trend and seasonal components
#         out = out + trend

#         # Project to output size
#         out = self.decoder(out[:, -1, :])

#         return out