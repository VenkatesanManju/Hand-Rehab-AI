"""
Multi-Head Recurrent Classification Models (Phase 3)
===================================================
Multi-head neural architectures for hand rehabilitation daily diary severity prediction.
Predicts 4 output heads:
- Pain (0-3)
- Stiffness (0-3)
- Mobility limitation (0-3)
- Grip limitation (0-3)

Encoders:
- Simple RNN
- LSTM
- GRU
- BiLSTM
- BiLSTM + Additive Attention (with token-level explainability weights)

Uses frozen pretrained GloVe 100d embeddings. Fixed seed: 42 (and 43, 44 for multi-seed eval).
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Additive / Bahdanau Self-Attention mechanism over token sequence hidden representations.
    Computes token attention scores and aggregates hidden states into a single context vector.
    Masks padding tokens to zero probability.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states: (batch_size, seq_len, hidden_dim)
        mask: (batch_size, seq_len), True for real tokens, False for padding
        Returns:
            context: (batch_size, hidden_dim)
            weights: (batch_size, seq_len)
        """
        # (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(self.projection(hidden_states))
        # (batch_size, seq_len)
        scores = self.context_vector(energy).squeeze(-1)

        if mask is not None:
            # Mask out padding tokens with a large negative value
            scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=-1)
        # Avoid NaN if sequence is entirely padding
        weights = torch.nan_to_num(weights, nan=0.0)

        # Context: weighted sum of hidden states
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class MultiHeadDiaryClassifier(nn.Module):
    """
    Multi-head sequence classifier for hand rehabilitation diaries.
    Shared encoder + 4 classification heads (pain, stiffness, mobility, grip).
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        encoder_type: str = "bilstm_attention",
        num_classes: int = 4,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = True,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.padding_idx = padding_idx
        self.hidden_dim = hidden_dim

        # 1. Embedding Layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=padding_idx,
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=padding_idx,
            )

        # 2. Recurrent Encoder
        if self.encoder_type == "rnn":
            self.encoder = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=1,
            )
            encoder_out_dim = hidden_dim
        elif self.encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=1,
            )
            encoder_out_dim = hidden_dim
        elif self.encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                num_layers=1,
            )
            encoder_out_dim = hidden_dim
        elif self.encoder_type == "bilstm":
            self.encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
            encoder_out_dim = hidden_dim * 2
        elif self.encoder_type in ("bilstm_attention", "bilstm+attention"):
            self.encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
            encoder_out_dim = hidden_dim * 2
            self.attention = AdditiveAttention(encoder_out_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.dropout = nn.Dropout(dropout)

        # 3. Four Output Classification Heads (Pain, Stiffness, Mobility, Grip)
        self.head_pain = nn.Linear(encoder_out_dim, num_classes)
        self.head_stiffness = nn.Linear(encoder_out_dim, num_classes)
        self.head_mobility = nn.Linear(encoder_out_dim, num_classes)
        self.head_grip = nn.Linear(encoder_out_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids: (batch_size, seq_len)
        lengths: (batch_size,) sequence lengths
        Returns dict of logits for each head, plus attention_weights if applicable.
        """
        # Mask for padding tokens: True for real tokens, False for padding
        mask = (input_ids != self.padding_idx)

        # Embedded: (batch_size, seq_len, embedding_dim)
        embeds = self.dropout(self.embedding(input_ids))

        # Pass through recurrent encoder
        out, _ = self.encoder(embeds)  # out: (batch_size, seq_len, encoder_out_dim)

        # Aggregate representations
        attention_weights = None
        if "attention" in self.encoder_type:
            pooled, attention_weights = self.attention(out, mask=mask)
        else:
            # Masked mean pooling over valid non-padded tokens
            expanded_mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            sum_hidden = (out * expanded_mask).sum(dim=1)
            valid_lengths = expanded_mask.sum(dim=1).clamp(min=1.0)
            pooled = sum_hidden / valid_lengths

        pooled = self.dropout(pooled)

        # Compute logits for each symptom head
        logits_pain = self.head_pain(pooled)
        logits_stiffness = self.head_stiffness(pooled)
        logits_mobility = self.head_mobility(pooled)
        logits_grip = self.head_grip(pooled)

        result = {
            "pain": logits_pain,
            "stiffness": logits_stiffness,
            "mobility": logits_mobility,
            "grip": logits_grip,
        }
        if attention_weights is not None:
            result["attention_weights"] = attention_weights

        return result


def compute_multihead_loss(
    logits_dict: Dict[str, torch.Tensor],
    targets_dict: Dict[str, torch.Tensor],
    head_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes total cross-entropy loss summed across the 4 symptom heads:
    L_total = L_pain + L_stiffness + L_mobility + L_grip
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, device=logits_dict["pain"].device)
    loss_breakdown: Dict[str, float] = {}

    symptoms = ["pain", "stiffness", "mobility", "grip"]
    for sym in symptoms:
        logits = logits_dict[sym]
        targets = targets_dict[sym]
        head_loss = criterion(logits, targets)
        w = head_weights.get(sym, 1.0) if head_weights else 1.0
        total_loss = total_loss + w * head_loss
        loss_breakdown[sym] = head_loss.item()

    return total_loss, loss_breakdown
