"""
PyTorch Dataset and DataLoader for Diary Severity Classification
================================================================
Converts preprocessed diary entries into PyTorch tensors with padding and collation.
"""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src.nlp.embeddings import GloVeEmbeddingLoader, PAD_IDX
from src.nlp.preprocess import preprocess_text


class DiaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_loader: GloVeEmbeddingLoader,
        max_len: int = 40,
    ):
        self.df = df.reset_index(drop=True)
        self.loader = embedding_loader
        self.max_len = max_len

        # Extract preprocessed tokens safely
        if "clean_tokens" in self.df.columns and not self.df["clean_tokens"].isna().all():
            self.token_lists = [str(t).split() if pd.notna(t) else [] for t in self.df["clean_tokens"]]
        else:
            self.token_lists = [
                preprocess_text(str(t), remove_stopwords=True, lemmatize=True, keep_negations=True)
                for t in self.df["text"]
            ]

        # Ensure no empty token lists (at least one token for RNN)
        for i in range(len(self.token_lists)):
            if len(self.token_lists[i]) == 0:
                self.token_lists[i] = ["symptom"]

        self.texts = self.df["text"].tolist()
        self.pain = self.df["pain"].values
        self.stiffness = self.df["stiffness"].values
        self.mobility = self.df["mobility"].values
        self.grip = self.df["grip"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tokens = self.token_lists[idx]
        length = min(len(tokens), self.max_len)
        input_ids = self.loader.tokens_to_indices(tokens, max_len=self.max_len, padding=True)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "pain": torch.tensor(self.pain[idx], dtype=torch.long),
            "stiffness": torch.tensor(self.stiffness[idx], dtype=torch.long),
            "mobility": torch.tensor(self.mobility[idx], dtype=torch.long),
            "grip": torch.tensor(self.grip[idx], dtype=torch.long),
            "text": self.texts[idx],
            "tokens": tokens,
        }


def collate_diary_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    lengths = torch.stack([item["length"] for item in batch])
    pain = torch.stack([item["pain"] for item in batch])
    stiffness = torch.stack([item["stiffness"] for item in batch])
    mobility = torch.stack([item["mobility"] for item in batch])
    grip = torch.stack([item["grip"] for item in batch])
    texts = [item["text"] for item in batch]
    tokens = [item["tokens"] for item in batch]

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "targets": {
            "pain": pain,
            "stiffness": stiffness,
            "mobility": mobility,
            "grip": grip,
        },
        "texts": texts,
        "tokens": tokens,
    }
