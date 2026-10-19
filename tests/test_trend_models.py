"""
Unit tests for Trend models and baselines (Phase 4)
"""

import pytest
import torch
import numpy as np
import pandas as pd
from src.trend.dataset import TrendWindowDataset, collate_trend_batch
from src.trend.models import MultiTaskTrendModel, compute_multitask_loss
from src.trend.baselines import TomorrowIsTodayBaseline, MovingAverageBaseline, RuleBasedStatusDetector
from src.trend.simulate import simulate_single_patient


def test_trend_window_dataset():
    rng = np.random.RandomState(42)
    df = simulate_single_patient("P001", rng, min_days=60, max_days=80)
    ds = TrendWindowDataset(df, window_size=14, fit_scaler=True)

    assert len(ds) == len(df) - 14
    sample = ds[0]
    assert sample["x"].shape == (14, 5)
    assert sample["raw_window"].shape == (14, 5)
    assert sample["y_forecast"].shape == (5,)
    assert sample["y_status"].item() in [0, 1, 2]


@pytest.mark.parametrize("encoder_type", ["rnn", "lstm", "gru"])
def test_multitask_trend_model_forward(encoder_type):
    model = MultiTaskTrendModel(
        input_dim=5,
        hidden_dim=32,
        encoder_type=encoder_type,
        num_layers=1,
    )
    x = torch.randn(8, 14, 5)
    forecast_out, status_logits = model(x)

    assert forecast_out.shape == (8, 5)
    assert status_logits.shape == (8, 3)

    target_norm = torch.randn(8, 5)
    target_cls = torch.randint(0, 3, (8,))
    total_loss, mse_val, ce_val = compute_multitask_loss(forecast_out, target_norm, status_logits, target_cls)
    assert total_loss.item() > 0.0


def test_trend_baselines():
    raw_windows = np.random.uniform(0.0, 3.0, size=(10, 14, 5))
    raw_windows[:, :, 4] = np.random.uniform(20.0, 80.0, size=(10, 14))

    # Persistence
    p_baseline = TomorrowIsTodayBaseline()
    pred_p = p_baseline.predict_forecast(raw_windows)
    assert pred_p.shape == (10, 5)
    np.testing.assert_allclose(pred_p, raw_windows[:, -1, :])

    # Moving Average
    ma_baseline = MovingAverageBaseline(ma_days=7)
    pred_ma = ma_baseline.predict_forecast(raw_windows)
    assert pred_ma.shape == (10, 5)
    np.testing.assert_allclose(pred_ma, np.mean(raw_windows[:, -7:, :], axis=1))

    # Rule-Based Detector
    detector = RuleBasedStatusDetector()
    status_preds = detector.predict_status(raw_windows)
    assert status_preds.shape == (10,)
    assert set(np.unique(status_preds)).issubset({0, 1, 2})
