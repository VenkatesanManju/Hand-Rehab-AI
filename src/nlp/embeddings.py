"""
Embeddings Loader & Vocab Builder
=================================
Loads pretrained GloVe 100d embeddings, builds vocabulary from preprocessed corpus,
constructs the PyTorch-compatible embedding matrix, and handles Out-Of-Vocabulary (OOV) words.

Fixed seed: 42
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_GLOVE_PATH = "data/embeddings/glove.6B/glove.6B.100d.txt"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


class GloVeEmbeddingLoader:
    def __init__(
        self,
        embedding_path: str = DEFAULT_GLOVE_PATH,
        embedding_dim: int = 100,
        seed: int = 42,
    ):
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.embedding_matrix: Optional[torch.Tensor] = None
        self.oov_words: Set[str] = set()
        self.coverage_stats: Dict[str, float] = {}

    def load_raw_embeddings(
        self,
        vocab_filter: Optional[Set[str]] = None,
        max_lines: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Loads word vectors from GloVe text file.
        If vocab_filter is provided, loads only vectors for words in vocab_filter
        to optimize memory and speed.
        """
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"GloVe embedding file not found at: {self.embedding_path}")

        embeddings: Dict[str, np.ndarray] = {}
        with open(self.embedding_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                parts = line.rstrip().split(" ")
                word = parts[0]
                if vocab_filter is None or word in vocab_filter:
                    vector = np.asarray(parts[1:], dtype=np.float32)
                    if len(vector) == self.embedding_dim:
                        embeddings[word] = vector

        return embeddings

    def build_vocab(
        self,
        tokenized_corpus: List[List[str]],
        min_freq: int = 1,
    ) -> Dict[str, int]:
        """
        Builds vocabulary from preprocessed corpus tokens.
        Index 0 is <pad>, Index 1 is <unk>.
        """
        freqs: Dict[str, int] = {}
        for doc in tokenized_corpus:
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1

        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

        # Sort by frequency then alphabetical for deterministic ordering
        sorted_tokens = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
        curr_idx = 2
        for tok, count in sorted_tokens:
            if count >= min_freq and tok not in self.word2idx:
                self.word2idx[tok] = curr_idx
                self.idx2word[curr_idx] = tok
                curr_idx += 1

        return self.word2idx

    def build_vocab_from_tokens(
        self,
        token_set: Set[str],
        glove_only: bool = True,
        raw_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, int]:
        """
        Builds vocabulary from a set of tokens across all splits.
        If glove_only is True, includes only tokens that exist in GloVe.
        Index 0 is <pad>, Index 1 is <unk>.
        """
        if glove_only and raw_embeddings is None:
            raw_embeddings = self.load_raw_embeddings(vocab_filter=token_set)

        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

        sorted_tokens = sorted(list(token_set))
        curr_idx = 2
        for tok in sorted_tokens:
            if glove_only and raw_embeddings is not None and tok not in raw_embeddings:
                continue
            if tok not in self.word2idx:
                self.word2idx[tok] = curr_idx
                self.idx2word[curr_idx] = tok
                curr_idx += 1

        return self.word2idx

    def build_embedding_matrix(
        self,
        raw_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> torch.Tensor:
        """
        Constructs the (vocab_size, embedding_dim) PyTorch tensor matrix.
        - <pad> is all zeros.
        - Known GloVe words copy the vector.
        - OOV words (including <unk>) are initialized from a normal distribution matching
          the mean and variance of the GloVe vectors (with fixed seed 42).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if raw_embeddings is None:
            # Load only words in vocabulary to be fast
            target_words = set(self.word2idx.keys())
            raw_embeddings = self.load_raw_embeddings(vocab_filter=target_words)

        # Estimate distribution statistics from loaded vectors
        if raw_embeddings:
            sample_vectors = np.array(list(raw_embeddings.values()))
            mean = sample_vectors.mean()
            std = sample_vectors.std()
        else:
            mean = 0.0
            std = 1.0

        vocab_size = len(self.word2idx)
        matrix = np.random.normal(mean, std, size=(vocab_size, self.embedding_dim)).astype(np.float32)

        # <pad> is zero vector
        matrix[PAD_IDX] = np.zeros(self.embedding_dim, dtype=np.float32)

        hits = 0
        self.oov_words = set()

        for word, idx in self.word2idx.items():
            if idx in (PAD_IDX, UNK_IDX):
                continue
            if word in raw_embeddings:
                matrix[idx] = raw_embeddings[word]
                hits += 1
            else:
                self.oov_words.add(word)

        # Total content words (excluding <pad> and <unk>)
        content_vocab_size = max(1, vocab_size - 2)
        coverage_pct = (hits / content_vocab_size) * 100.0

        self.coverage_stats = {
            "vocab_size": vocab_size,
            "content_words": content_vocab_size,
            "glove_hits": hits,
            "oov_count": len(self.oov_words),
            "coverage_pct": round(coverage_pct, 2),
        }

        self.embedding_matrix = torch.tensor(matrix, dtype=torch.float32)
        return self.embedding_matrix

    def tokens_to_indices(
        self,
        tokens: List[str],
        max_len: Optional[int] = None,
        padding: bool = True,
    ) -> List[int]:
        """Converts token list to index list with optional truncation and padding."""
        indices = [self.word2idx.get(tok, UNK_IDX) for tok in tokens]
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            elif padding and len(indices) < max_len:
                indices = indices + [PAD_IDX] * (max_len - len(indices))
        return indices

    def batch_to_tensor(
        self,
        batch_tokens: List[List[str]],
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts batch of token lists to:
        - input_ids tensor: (batch_size, max_seq_len)
        - lengths tensor: (batch_size,)
        """
        lengths = [len(toks) for toks in batch_tokens]
        if max_len is None:
            max_len = max(lengths) if lengths else 1

        all_ids = [self.tokens_to_indices(toks, max_len=max_len, padding=True) for toks in batch_tokens]
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        lengths_tensor = torch.tensor([min(l, max_len) for l in lengths], dtype=torch.long)
        return input_ids, lengths_tensor

    def save_vocab_and_matrix(self, vocab_path: str, matrix_path: str) -> None:
        """Saves vocabulary mapping and embedding matrix tensor."""
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, indent=2)

        if self.embedding_matrix is not None:
            torch.save(self.embedding_matrix, matrix_path)

    def load_vocab_and_matrix(self, vocab_path: str, matrix_path: str) -> None:
        """Loads vocabulary mapping and embedding matrix tensor."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.embedding_matrix = torch.load(matrix_path, weights_only=True)
