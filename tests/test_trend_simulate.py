"""
Unit tests for longitudinal recovery curves simulation (Phase 4)
"""

import pytest
import numpy as np
import pandas as pd
from src.trend.simulate import (
    simulate_single_patient,
    generate_simulated_dataset,
    split_by_patient,
    FEATURE_COLS,
)


def test_simulate_single_patient_structure():
    rng = np.random.RandomState(42)
    df = simulate_single_patient("P001", rng, min_days=60, max_days=120)

    assert len(df) >= 60 and len(df) <= 120
    expected_cols = ["patient_id", "day", "pain", "stiffness", "mobility", "grip", "rom", "status", "status_label", "is_simulated"]
    for col in expected_cols:
        assert col in df.columns

    # Verify patient_id and day index
    assert (df["patient_id"] == "P001").all()
    assert (df["day"] == np.arange(len(df))).all()
    assert (df["is_simulated"] == True).all()


def test_feature_bounds_and_validity():
    rng = np.random.RandomState(123)
    df = simulate_single_patient("P002", rng, min_days=60, max_days=120, is_noisy_setting=True)

    for sym in ["pain", "stiffness", "mobility", "grip"]:
        assert (df[sym] >= 0.0).all(), f"{sym} below 0.0"
        assert (df[sym] <= 3.0).all(), f"{sym} above 3.0"

    assert (df["rom"] >= 0.0).all(), "rom below 0.0"
    assert (df["rom"] <= 100.0).all(), "rom above 100.0"

    assert set(df["status"].unique()).issubset({0, 1, 2})
    assert set(df["status_label"].unique()).issubset({"normal", "plateau", "relapse"})


def test_split_by_patient_disjointness():
    df = generate_simulated_dataset(num_patients=30, seed=42)
    train_df, val_df, test_df = split_by_patient(df, train_ratio=0.7, val_ratio=0.15, seed=42)

    train_pids = set(train_df["patient_id"].unique())
    val_pids = set(val_df["patient_id"].unique())
    test_pids = set(test_df["patient_id"].unique())

    # Patient sets must be strictly mutually exclusive
    assert len(train_pids & val_pids) == 0, "Train and Val share patient IDs"
    assert len(train_pids & test_pids) == 0, "Train and Test share patient IDs"
    assert len(val_pids & test_pids) == 0, "Val and Test share patient IDs"

    # All patients accounted for
    assert train_pids | val_pids | test_pids == set(df["patient_id"].unique())
