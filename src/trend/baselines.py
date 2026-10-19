"""
Classical & Heuristic Baselines for Trend Recovery (Phase 4)
============================================================
1. "Tomorrow = Today" (Persistence Forecast)
2. 7-Day Moving Average Forecast
3. Rule-Based Plateau & Relapse Detector (Slope Thresholds)
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support


class TomorrowIsTodayBaseline:
    """
    Naive persistence forecast: predicts tomorrow's value equals today's value (last day of window).
    """
    def __init__(self):
        self.name = "Tomorrow = Today"

    def predict_forecast(self, raw_windows: np.ndarray) -> np.ndarray:
        # raw_windows: (N, 14, 5) -> take last step (day 13)
        return raw_windows[:, -1, :]

    def predict_status(self, raw_windows: np.ndarray) -> np.ndarray:
        # Predict majority class (0: normal)
        return np.zeros(len(raw_windows), dtype=int)


class MovingAverageBaseline:
    """
    7-Day Moving Average Forecast: predicts tomorrow's value as the mean of the past 7 days.
    """
    def __init__(self, ma_days: int = 7):
        self.name = f"{ma_days}-Day Moving Average"
        self.ma_days = ma_days

    def predict_forecast(self, raw_windows: np.ndarray) -> np.ndarray:
        # raw_windows: (N, 14, 5) -> average over last ma_days
        return np.mean(raw_windows[:, -self.ma_days :, :], axis=1)

    def predict_status(self, raw_windows: np.ndarray) -> np.ndarray:
        return np.zeros(len(raw_windows), dtype=int)


class RuleBasedStatusDetector:
    """
    Rule-based Plateau & Relapse detector based on slope thresholds over the past 7 days.
    - Relapse: symptoms worsening rapidly (positive slope) or ROM dropping rapidly
    - Plateau: progress stalled (|slope| <= epsilon)
    - Normal: steady expected improvement (symptoms negative slope, ROM positive slope)
    """
    def __init__(
        self,
        lookback_days: int = 7,
        relapse_sym_threshold: float = 0.02,
        relapse_rom_threshold: float = 0.4,
        plateau_sym_threshold: float = 0.012,
        plateau_rom_threshold: float = 0.25,
    ):
        self.name = "Rule-Based Detector"
        self.lookback_days = lookback_days
        self.relapse_sym_threshold = relapse_sym_threshold
        self.relapse_rom_threshold = relapse_rom_threshold
        self.plateau_sym_threshold = plateau_sym_threshold
        self.plateau_rom_threshold = plateau_rom_threshold

    def tune_on_validation(self, raw_windows: np.ndarray, y_true_status: np.ndarray):
        """
        Tunes detection thresholds strictly on validation data to maximize macro F1.
        """
        best_f1 = -1.0
        best_params = (self.relapse_sym_threshold, self.plateau_sym_threshold)

        for r_sym in [0.015, 0.020, 0.030, 0.040]:
            for p_sym in [0.005, 0.008, 0.012, 0.018]:
                self.relapse_sym_threshold = r_sym
                self.plateau_sym_threshold = p_sym
                preds = self.predict_status(raw_windows)
                _, _, f1, _ = precision_recall_fscore_support(
                    y_true_status, preds, average="macro", zero_division=0
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (r_sym, p_sym)

        self.relapse_sym_threshold, self.plateau_sym_threshold = best_params

    def predict_status(self, raw_windows: np.ndarray) -> np.ndarray:
        # raw_windows: (N, 14, 5)
        # Features 0..3: symptoms (pain, stiffness, mobility, grip)
        # Feature 4: ROM
        recent = raw_windows[:, -self.lookback_days :, :]  # (N, lookback, 5)
        n_samples = len(raw_windows)
        preds = np.zeros(n_samples, dtype=int)

        # Average daily change over lookback days: (last - first) / (lookback - 1)
        sym_diff = np.mean(recent[:, -1, :4] - recent[:, 0, :4], axis=1) / (self.lookback_days - 1)
        rom_diff = (recent[:, -1, 4] - recent[:, 0, 4]) / (self.lookback_days - 1)

        for i in range(n_samples):
            sd = sym_diff[i]
            rd = rom_diff[i]

            # Condition 1: Relapse (symptoms worsening or ROM deteriorating)
            if sd > self.relapse_sym_threshold or rd < -self.relapse_rom_threshold:
                preds[i] = 2  # Relapse
            # Condition 2: Plateau (symptoms and ROM flat / stalled)
            elif abs(sd) <= self.plateau_sym_threshold and abs(rd) <= self.plateau_rom_threshold:
                preds[i] = 1  # Plateau
            else:
                preds[i] = 0  # Normal recovery

        return preds
