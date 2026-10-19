"""
Unit Tests for Embeddings & Vocabulary Module (Phase 2)
======================================================
Tests GloVe loader, vocab builder, embedding matrix shape,
padding token vectors, and tensor batch conversion.
"""

import os
import json
import pytest
import torch
import numpy as np

from src.nlp.embeddings import (
    GloVeEmbeddingLoader,
    PAD_TOKEN,
    UNK_TOKEN,
    PAD_IDX,
    UNK_IDX,
    DEFAULT_GLOVE_PATH,
)


def test_vocab_structure():
    """Vocabulary correctly assigns PAD_IDX=0, UNK_IDX=1 and counts tokens."""
    corpus = [
        ["no", "pain", "today"],
        ["stiff", "finger", "morning"],
        ["pain", "free"],
    ]
    loader = GloVeEmbeddingLoader(embedding_path=DEFAULT_GLOVE_PATH, embedding_dim=100)
    word2idx = loader.build_vocab(corpus, min_freq=1)

    assert word2idx[PAD_TOKEN] == PAD_IDX
    assert word2idx[UNK_TOKEN] == UNK_IDX
    assert "pain" in word2idx
    assert "no" in word2idx
    # 'pain' appears twice, so it should have a low index
    assert word2idx["pain"] == 2


def test_tokens_to_indices():
    """Mapping tokens to indices handles known, OOV, padding, and truncation."""
    corpus = [["no", "pain", "stiff"]]
    loader = GloVeEmbeddingLoader(embedding_path=DEFAULT_GLOVE_PATH, embedding_dim=100)
    loader.build_vocab(corpus)

    # Known words
    indices = loader.tokens_to_indices(["no", "pain"])
    assert indices == [loader.word2idx["no"], loader.word2idx["pain"]]

    # Unknown word mapped to UNK_IDX
    oov_indices = loader.tokens_to_indices(["completely_unknown_token_xyz"])
    assert oov_indices == [UNK_IDX]

    # Truncation
    trunc = loader.tokens_to_indices(["no", "pain", "stiff"], max_len=2)
    assert len(trunc) == 2

    # Padding
    padded = loader.tokens_to_indices(["no"], max_len=4, padding=True)
    assert len(padded) == 4
    assert padded[1:] == [PAD_IDX, PAD_IDX, PAD_IDX]


def test_batch_to_tensor():
    """Batch tensor creation returns correctly sized LongTensors."""
    corpus = [["no", "pain"], ["finger", "very", "stiff", "today"]]
    loader = GloVeEmbeddingLoader(embedding_path=DEFAULT_GLOVE_PATH, embedding_dim=100)
    loader.build_vocab(corpus)

    batch = [["no", "pain"], ["stiff"]]
    input_ids, lengths = loader.batch_to_tensor(batch, max_len=5)

    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.dtype == torch.long
    assert input_ids.shape == (2, 5)
    assert lengths.tolist() == [2, 1]
    assert input_ids[1, 1].item() == PAD_IDX


def test_embedding_matrix_properties():
    """Embedding matrix has shape (vocab_size, 100) and pad vector is all zeros."""
    matrix_path = "data/processed/glove_100d_matrix.pt"
    vocab_path = "data/processed/vocab.json"

    assert os.path.exists(matrix_path), "Embedding matrix file missing!"
    assert os.path.exists(vocab_path), "Vocab JSON missing!"

    with open(vocab_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    matrix = torch.load(matrix_path, weights_only=True)
    assert matrix.shape == (len(word2idx), 100)
    # Pad vector at index 0 must be strictly zeros
    assert torch.all(matrix[PAD_IDX] == 0.0)
    # Unknown vector at index 1 must not be all zeros
    assert not torch.all(matrix[UNK_IDX] == 0.0)
