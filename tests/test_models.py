"""
Unit Tests for Phase 3 Multi-Head Models & Baseline
===================================================
Tests forward pass shapes, loss computation, attention weight normalisation,
and TF-IDF baseline training/evaluation.
"""

import pytest
import torch
import numpy as np

from src.nlp.models import MultiHeadDiaryClassifier, compute_multihead_loss
from src.nlp.baseline import TfidfLogRegBaseline


@pytest.mark.parametrize("encoder", ["rnn", "lstm", "gru", "bilstm", "bilstm_attention"])
def test_multihead_classifier_forward(encoder):
    """All 5 encoder types produce valid logits for the 4 symptom heads."""
    vocab_size = 50
    embedding_dim = 16
    hidden_dim = 32
    batch_size = 4
    seq_len = 10

    model = MultiHeadDiaryClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        encoder_type=encoder,
        num_classes=4,
        freeze_embeddings=False,
    )
    model.eval()

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([10, 8, 6, 4])

    outputs = model(input_ids, lengths=lengths)

    for sym in ["pain", "stiffness", "mobility", "grip"]:
        assert sym in outputs, f"Missing output head: {sym}"
        assert outputs[sym].shape == (batch_size, 4), f"Wrong shape for {sym}: {outputs[sym].shape}"

    if "attention" in encoder:
        assert "attention_weights" in outputs
        weights = outputs["attention_weights"]
        assert weights.shape == (batch_size, seq_len)
        # Weights should sum to approximately 1.0 per sequence
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_multihead_loss_computation():
    """Multi-head loss correctly sums cross-entropy across all 4 symptom heads."""
    batch_size = 4
    logits = {
        sym: torch.randn(batch_size, 4, requires_grad=True)
        for sym in ["pain", "stiffness", "mobility", "grip"]
    }
    targets = {
        sym: torch.randint(0, 4, (batch_size,))
        for sym in ["pain", "stiffness", "mobility", "grip"]
    }

    total_loss, breakdown = compute_multihead_loss(logits, targets)
    assert total_loss.item() > 0.0
    assert len(breakdown) == 4
    for sym in ["pain", "stiffness", "mobility", "grip"]:
        assert sym in breakdown
        assert breakdown[sym] > 0.0

    total_loss.backward()
    for sym in logits:
        assert logits[sym].grad is not None


def test_tfidf_baseline():
    """TF-IDF baseline fits and predicts for all 4 symptom dimensions."""
    texts = [
        "no pain today",
        "sharp pain in wrist",
        "stiff knuckles morning",
        "cannot grip cups easily",
    ]
    targets = {
        "pain": np.array([0, 3, 0, 1]),
        "stiffness": np.array([0, 0, 3, 0]),
        "mobility": np.array([0, 1, 2, 0]),
        "grip": np.array([0, 2, 0, 3]),
    }

    baseline = TfidfLogRegBaseline(random_state=42)
    baseline.fit(texts, targets)
    preds = baseline.predict(texts)

    for sym in ["pain", "stiffness", "mobility", "grip"]:
        assert len(preds[sym]) == len(texts)
        assert set(preds[sym]).issubset({0, 1, 2, 3})

    metrics = baseline.evaluate(texts, targets)
    assert "overall" in metrics
    assert "mean_macro_f1" in metrics["overall"]
