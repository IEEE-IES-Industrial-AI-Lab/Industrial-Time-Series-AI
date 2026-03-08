import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    Standard LSTM for multivariate time series forecasting and anomaly detection.
    """
    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout=0.1, out_features=None):
        """
        Args:
            num_features (int): Number of input variables (sensors).
            hidden_dim (int): LSTM hidden state dimension.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability between LSTM layers.
            out_features (int): Dimension of the output prediction. Defaults to num_features.
        """
        super(LSTMForecaster, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features if out_features is not None else num_features
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Typically maps the last hidden state to the desired output dimension
        self.fc = nn.Linear(hidden_dim, self.out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, window_size, num_features)
        Returns:
            torch.Tensor: Shape (batch_size, out_features)
        """
        # lstm_out shape: (batch_size, window_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        
        predictions = self.fc(last_out)
        return predictions

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for unsupervised anomaly detection.
    Reconstructs the input sequence. High reconstruction error -> Anomaly.
    """
    def __init__(self, num_features, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, num_features, num_layers, batch_first=True)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, window_size, num_features)
        Returns:
            torch.Tensor: Reconstructed x
        """
        # Encode
        _, (h_n, c_n) = self.encoder(x)
        
        # Repeat the last hidden state for the decoder
        # shape: (batch_size, window_size, hidden_dim)
        seq_len = x.shape[1]
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        reconstruction, _ = self.decoder(decoder_input)
        return reconstruction
