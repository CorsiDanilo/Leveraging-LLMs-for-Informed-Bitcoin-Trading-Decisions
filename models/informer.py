# import torch
# from torch import nn
# import numpy as np
# from models.utils.models_utils import process_embeddings

# # Select device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# class ProbSparseAttention(nn.Module):
#     def __init__(self, model_dim, num_heads):
#         super(ProbSparseAttention, self).__init__()
#         self.model_dim = model_dim
#         self.num_heads = num_heads
#         self.q_linear = nn.Linear(model_dim, model_dim)
#         self.k_linear = nn.Linear(model_dim, model_dim)
#         self.v_linear = nn.Linear(model_dim, model_dim)
#         self.out = nn.Linear(model_dim, model_dim)
        
#         self.dropout = nn.Dropout(p=0.1)
#         self.scale = torch.sqrt(torch.FloatTensor([model_dim // num_heads])).to(device)

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
        
#         Q = self.q_linear(query)
#         K = self.k_linear(key)
#         V = self.v_linear(value)
        
#         Q = Q.view(batch_size, -1, self.num_heads, self.model_dim // self.num_heads)
#         K = K.view(batch_size, -1, self.num_heads, self.model_dim // self.num_heads)
#         V = V.view(batch_size, -1, self.num_heads, self.model_dim // self.num_heads)
        
#         energy = torch.einsum("bqhd,bkhd->bhqk", Q, K) / self.scale
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, -1e10)
        
#         attention = torch.nn.functional.softmax(energy, dim=-1)
#         attention = self.dropout(attention)
        
#         x = torch.einsum("bhql,blhd->bqhd", attention, V).contiguous()
#         x = x.view(batch_size, -1, self.model_dim)
        
#         return self.out(x)

# class InformerEncoderLayer(nn.Module):
#     def __init__(self, model_dim, num_heads, dropout=0.1):
#         super(InformerEncoderLayer, self).__init__()
#         self.prob_sparse_attention = ProbSparseAttention(model_dim, num_heads)
#         self.norm1 = nn.LayerNorm(model_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(model_dim, model_dim * 4),
#             nn.ReLU(),
#             nn.Linear(model_dim * 4, model_dim)
#         )
#         self.norm2 = nn.LayerNorm(model_dim)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, src, src_mask=None):
#         src2 = self.prob_sparse_attention(src, src, src, src_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.ffn(src)
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

# class Informer(nn.Module):
#     def __init__(self, parameters):
#         super(Informer, self).__init__()
#         self.embedding_layers = parameters['embedding_layers']
#         self.embedding_dim = parameters['embedding_dim']
#         self.num_continuous_features = parameters['num_continuous_features']
#         self.model_dim = parameters['model_dim']
#         self.num_heads = parameters['num_heads']
#         self.num_layers = parameters['num_layers']
#         self.output_size = parameters['output_size']
#         self.dropout = parameters['dropout']

#         self.encoder = nn.Linear(self.num_continuous_features + self.embedding_dim, self.model_dim)
#         self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
#         self.encoder_layers = nn.ModuleList([InformerEncoderLayer(self.model_dim, self.num_heads, self.dropout) for _ in range(self.num_layers)])
#         self.decoder = nn.Linear(self.model_dim, self.output_size)

#     def forward(self, x_continuous, x_categorical):
#         # Process embeddings for categorical features
#         x_combined = process_embeddings(self.embedding_layers, x_categorical, x_continuous)

#         out = self.encoder(x_combined)
#         out = self.pos_encoder(out)
#         for layer in self.encoder_layers:
#             out = layer(out)
#         out = self.decoder(out[:, -1, :])
#         return out

from torch import nn
import torch
import numpy as np
from models.utils.models_utils import process_embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(ProbSparseSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.model_dim // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.model_dim // self.num_heads)
        attn_scores = self.dropout(torch.softmax(attn_scores, dim=-1))
        attn_output = torch.matmul(attn_scores, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        return attn_output

class InformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(InformerEncoderLayer, self).__init__()
        self.self_attn = ProbSparseSelfAttention(model_dim, num_heads, dropout)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout(nn.ReLU()(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_output))
        return x

class InformerEncoder(nn.Module):
    def __init__(self, model_dim, num_heads, num_layers, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(model_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Informer(nn.Module):
    def __init__(self, parameters, dropout=0.1):
        super(Informer, self).__init__()
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
        self.encoder_layers = InformerEncoder(self.model_dim, self.num_heads, self.num_layers, self.dropout)
        self.decoder = nn.Linear(self.model_dim, self.output_size)

    def forward(self, x_continuous, x_categorical):
        x_combined = process_embeddings(self.embedding_layers, x_categorical, x_continuous)
        out = self.encoder(x_combined)
        out = self.pos_encoder(out)
        out = self.encoder_layers(out)
        out = self.decoder(out[:, -1, :])
        return out
