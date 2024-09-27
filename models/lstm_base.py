from torch import nn
import torch
from models.utils.models_utils import process_embeddings

class LSTMBase(nn.Module):
    """ 
    LSTM model with embeddings for categorical features and continuous features 
    
    Parameters:
        embedding_layers (list): List of embedding layers for categorical features
        num_continuous_features (int): Number of continuous features
        embedding_dim (int): Dimension of embeddings
        hidden_size (int): Hidden size of LSTM
        num_layers (int): Number of LSTM layers
        output_size (int): Output size
        dropout (float): Dropout rate

    Returns:
        out (torch.Tensor): Output tensor
    """

    def __init__(self, parameters, dropout=0.1):
        super(LSTMBase, self).__init__()
        self.embedding_layers = parameters['embedding_layers']
        self.num_continuous_features = parameters['num_continuous_features']
        self.embedding_dim = parameters['embedding_dim']
        self.hidden_size = parameters['hidden_size']
        self.num_layers = parameters['num_layers']
        self.output_size = parameters['output_size']
        self.dropout = parameters['dropout']    
        
        # Define the lstm layer
        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            input_size=self.num_continuous_features + self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            batch_first=True,
        )

        # Define the fully connected layer
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x_continuous, x_categorical):
        # Process embeddings for categorical features
        x_combined = process_embeddings(self.embedding_layers, x_categorical, x_continuous)

        # Pass through LSTM layer
        out, _ = self.lstm(x_combined.unsqueeze(1))
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.linear(out)
        
        return out