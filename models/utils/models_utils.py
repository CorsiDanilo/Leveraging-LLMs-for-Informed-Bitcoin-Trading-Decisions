import torch

def process_embeddings(embedding_layers, x_categorical, x_continuous):
    """ Process embeddings for categorical features and concatenate with continuous features.

    Args:
        embedding_layers (dict): Dictionary containing the embedding layers for each categorical feature.
        x_categorical (dict): Dictionary containing the categorical features.
        x_continuous (torch.Tensor): Tensor containing the continuous features.

    Returns:
        torch.Tensor: Tensor containing the processed embeddings and continuous features.
    """
    # Process embeddings for categorical features
    embeddings = []
    for key, value in x_categorical.items():
        emb = embedding_layers[key](value)
        embeddings.append(emb)
    
    # Stack all embeddings
    x_categorical_combined = torch.stack(embeddings, dim=-1)
    
    # Sum embeddings along the last dimension
    x_categorical_combined = torch.sum(x_categorical_combined, dim=-1)
    
    # Concatenate with continuous features along the last dimension
    x_combined = torch.cat([x_continuous, x_categorical_combined], dim=-1)

    return x_combined