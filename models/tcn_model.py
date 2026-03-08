import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Ensures causal convolution by removing the extra padding at the end.
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN).
    Excellent for parallelism and capturing very long receptive fields efficiently.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs (int): Number of input features.
            num_channels (list): List of channel sizes for each layer.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, num_features, seq_len) -> NOTE: diff from standard (B, L, C)
        Returns:
            torch.Tensor: Shape (batch_size, out_channels[-1], seq_len)
        """
        return self.network(x)

class TCNForecaster(nn.Module):
    def __init__(self, num_features, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2, out_features=None):
        super(TCNForecaster, self).__init__()
        self.tcn = TemporalConvNet(num_features, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.out_features = out_features if out_features is not None else num_features
        self.fc = nn.Linear(num_channels[-1], self.out_features)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, num_features)
        """
        # TCN expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        
        # y1 shape: (batch_size, out_channel, seq_len)
        y1 = self.tcn(x)
        
        # Take the last prediction
        out = self.fc(y1[:, :, -1])
        return out
