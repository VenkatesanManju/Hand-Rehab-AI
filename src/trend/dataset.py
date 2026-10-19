"""
PyTorch Dataset and Sliding Window Generator for Trend Recovery Sequences (Phase 4)
===================================================================================
SIMULATED DATA ONLY - NO CLINICAL VALIDITY.

Extracts fixed 14-day sliding windows per patient:
- Input X: (14, 5) features: [pain, stiffness, mobility, grip, rom]
- Target Forecast: (5,) continuous values at day t: [pain, stiffness, mobility, grip, rom]
- Target Status: integer class in {0: normal, 1: plateau, 2: relapse} at day t
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

FEATURE_COLS = ["pain", "stiffness", "mobility", "grip", "rom"]


class TrendWindowDataset(Dataset):
    """
    Sliding window dataset over patient trajectories.
    Extracts windows within each patient (never spanning across different patients).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 14,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None,
        fit_scaler: bool = False,
    ):
        self.window_size = window_size
        self.df = df.copy()

        # Compute or set scaler parameters
        raw_features = self.df[FEATURE_COLS].values.astype(np.float32)
        if fit_scaler:
            self.scaler_mean = np.mean(raw_features, axis=0)
            self.scaler_std = np.std(raw_features, axis=0) + 1e-6
        else:
            self.scaler_mean = scaler_mean if scaler_mean is not None else np.zeros(5, dtype=np.float32)
            self.scaler_std = scaler_std if scaler_std is not None else np.ones(5, dtype=np.float32)

        self.samples = []
        self._build_windows()

    def _build_windows(self):
        grouped = self.df.groupby("patient_id")
        for pid, group in grouped:
            group = group.sort_values("day").reset_index(drop=True)
            feats = group[FEATURE_COLS].values.astype(np.float32)
            statuses = group["status"].values.astype(np.int64)
            n_days = len(group)

            if n_days <= self.window_size:
                continue

            for t in range(self.window_size, n_days):
                # Window: day t - window_size to day t - 1
                raw_window = feats[t - self.window_size : t]
                # Normalize window
                norm_window = (raw_window - self.scaler_mean) / self.scaler_std

                # Targets at day t
                target_forecast = feats[t]  # Real unscaled physical values for metric computation
                target_status = statuses[t]

                self.samples.append({
                    "patient_id": pid,
                    "target_day": t,
                    "window": norm_window.astype(np.float32),
                    "raw_window": raw_window.astype(np.float32),
                    "target_forecast": target_forecast.astype(np.float32),
                    "target_status": target_status,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "x": torch.tensor(s["window"], dtype=torch.float32),
            "raw_window": torch.tensor(s["raw_window"], dtype=torch.float32),
            "y_forecast": torch.tensor(s["target_forecast"], dtype=torch.float32),
            "y_status": torch.tensor(s["target_status"], dtype=torch.long),
            "patient_id": s["patient_id"],
            "target_day": s["target_day"],
        }


def collate_trend_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([item["x"] for item in batch])
    raw_window = torch.stack([item["raw_window"] for item in batch])
    y_forecast = torch.stack([item["y_forecast"] for item in batch])
    y_status = torch.stack([item["y_status"] for item in batch])
    patient_ids = [item["patient_id"] for item in batch]
    target_days = [item["target_day"] for item in batch]

    return {
        "x": x,
        "raw_window": raw_window,
        "y_forecast": y_forecast,
        "y_status": y_status,
        "patient_ids": patient_ids,
        "target_days": target_days,
    }
