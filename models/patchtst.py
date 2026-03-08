import torch
import torch.nn as nn
from .transformer_ts import PositionalEncoding

class PatchingLayer(nn.Module):
    """
    Converts a time-series into patches. Given sequence length L, breaks
    it into patches of size P with stride S.
    """
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features, seq_len)
        Returns:
            patches: (batch_size, num_features, num_patches, patch_len)
        """
        # Unfold extracts sliding local blocks from a batched input tensor.
        # Here we apply it to the 1D sequence dimension
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return patches

class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words (SOTA 2023).
    Key features: 
    1. Channel Independence (each sensor processed independently).
    2. Patching (reduces sequence length, extracts local semantics).
    """
    def __init__(self, num_features, seq_len, pred_len, patch_len=16, stride=8, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super(PatchTST, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.patching = PatchingLayer(patch_len, stride)
        
        # Calculate number of patches
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        
        # Linear projection of patch
        self.W_P = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Final flattening and projection to prediction length
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_patches * d_model, pred_len)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, num_features]
        Returns:
            Tensor, shape [batch_size, pred_len, num_features]
        """
        b, l, c = x.shape
        
        # Channel Independence: treat features as batch elements
        # Reshape to (batch_size * num_features, 1, seq_len)
        x = x.permute(0, 2, 1).contiguous()
        x_flat = x.view(b * c, 1, l)
        
        # Patching -> (b*c, 1, num_patches, patch_len)
        patches = self.patching(x_flat)
        patches = patches.squeeze(1) # (b*c, num_patches, patch_len)
        
        # Projection -> (b*c, num_patches, d_model)
        x_emb = self.W_P(patches)
        x_emb = self.pos_encoder(x_emb)
        
        # Transformer
        out = self.transformer(x_emb) # (b*c, num_patches, d_model)
        
        # Head -> (b*c, pred_len)
        preds = self.head(out)
        
        # Reshape back to (batch_size, pred_len, num_features)
        preds = preds.view(b, c, self.pred_len).permute(0, 2, 1)
        
        return preds
