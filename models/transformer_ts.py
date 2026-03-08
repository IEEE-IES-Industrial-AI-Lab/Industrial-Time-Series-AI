import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
            
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Standard Transformer Encoder architecture adapted for Time-Series Forecasting.
    """
    def __init__(self, num_features, d_model=64, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1, out_features=None):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.out_features = out_features if out_features is not None else num_features
        
        # Feature embedder
        self.feature_extractor = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, self.out_features)
        
    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, num_features]
        Returns:
            Tensor, shape [batch_size, out_features]
        """
        # Embed and add positional encoding
        src = self.feature_extractor(src)
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        
        # We take the output representing the last time step to predict the future
        output = output[:, -1, :]
        
        output = self.decoder(output)
        return output
