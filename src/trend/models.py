"""
Recurrent Multi-Task Architectures for Recovery Trend Modeling (Phase 4)
========================================================================
Encoders: Simple RNN, LSTM, GRU.
Input: (batch_size, 14, 5) [pain, stiffness, mobility, grip, rom]
Heads:
1. Forecast Head: outputs (batch_size, 5) predicting day t+1 values
2. Classification Head: outputs (batch_size, 3) predicting status (normal, plateau, relapse)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskTrendModel(nn.Module):
    """
    Multi-task recurrent neural network for hand rehabilitation recovery trajectories.
    Shared encoder + Forecast Head + Plateau/Relapse Classification Head.
    """
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        encoder_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.2,
        num_status_classes: int = 3,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 1. Recurrent Encoder
        if self.encoder_type == "rnn":
            self.encoder = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        self.dropout = nn.Dropout(dropout)

        # 2. Next-Day Forecast Head (5 continuous features)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, input_dim),
        )

        # 3. Status Classification Head (3 classes: normal, plateau, relapse)
        self.status_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_status_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len=14, input_dim=5)
        Returns:
            forecast_out: (batch_size, 5)
            status_logits: (batch_size, 3)
        """
        out, h_n = self.encoder(x)

        # Extract last hidden representation
        if isinstance(h_n, tuple):  # LSTM (h_n, c_n)
            rep = h_n[0][-1]  # (batch_size, hidden_dim)
        else:
            rep = h_n[-1]  # (batch_size, hidden_dim)

        rep = self.dropout(rep)

        forecast_out = self.forecast_head(rep)
        status_logits = self.status_head(rep)

        return forecast_out, status_logits


def compute_multitask_loss(
    pred_forecast: torch.Tensor,
    target_forecast_norm: torch.Tensor,
    pred_status_logits: torch.Tensor,
    target_status: torch.Tensor,
    forecast_weight: float = 1.0,
    cls_weight: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Computes joint loss: MSE (forecast) + CrossEntropy (status classification).
    """
    mse_loss = F.mse_loss(pred_forecast, target_forecast_norm)
    ce_loss = F.cross_entropy(pred_status_logits, target_status)
    total_loss = forecast_weight * mse_loss + cls_weight * ce_loss
    return total_loss, mse_loss.item(), ce_loss.item()
