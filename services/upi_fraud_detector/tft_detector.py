"""
TFT-based UPI Fraud Detector — STUB for Hackathon MVP.

This module would implement a Temporal Fusion Transformer for sequence modeling
of 30-day transaction history per merchant/UPI handle.

Purpose in production:
  - Models temporal patterns in transaction sequences
  - Handles variable-length history with attention
  - Captures long-range dependencies (e.g., gradual velocity buildup)
  - Provides interpretability via attention weights

Why NOT implemented for this hackathon build:
  - Requires GPU/CUDA for reasonable training time
  - Adds significant complexity (sequence preprocessing, padding, masking)
  - LightGBM on engineered features achieves comparable results for MVP
  - TFT shines with longer histories and more complex temporal patterns

To enable in production:
  1. Implement sequence feature extraction (30-day windows per entity)
  2. Add TFT model with PyTorch (requires GPU)
  3. Implement sequence padding/collation for batch training
  4. Requires ~4-6 weeks of ML engineering + GPU infrastructure
"""


import torch
from torch import nn


class TemporalFusionTransformer(nn.Module):
    """
    Placeholder for TFT model.
    
    In production, this would:
      - Accept (batch, seq_len, n_features) input
      - Use variable selection networks
      - Apply multi-head attention over time
      - Output risk score per sequence
    
    For hackathon: NOT USED — replaced by LightGBM on static features.
    """

    def __init__(self, input_dim: int = 100, hidden_size: int = 256,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Minimal stub architecture
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, sequence, static):
        """STUB: Just uses static features, ignores sequence."""
        return torch.sigmoid(self.fc(static.mean(dim=1)))


class TFTUpiFraudDetector:
    """STUB wrapper matching LightGBM detector interface."""

    def __init__(self, config=None):
        self.config = config
        self.model = TemporalFusionTransformer()
        self.is_fitted = False

    def train(self, X, y, R=None, O=None, D=None, groups=None):
        print("[TFT] Skipped — using LightGBM for hackathon MVP")
        self.is_fitted = True
        return {"pr_auc": 0.0, "roc_auc": 0.0}

    def predict(self, features):
        return [{"score": 0.0, "action": "ALLOW", "reasons": ["TFT not implemented"], "shap_top3": []}]

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()


# Export for compatibility
__all__ = ["TFTUpiFraudDetector", "TemporalFusionTransformer"]
