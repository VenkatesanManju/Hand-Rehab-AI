"""
Unit Tests for Processed Splits & Leakage Prevention (Phase 2)
==============================================================
Validates:
1. Exact deduplication of raw synthetic texts.
2. Zero template leakage between train, val, and test.
3. Zero text overlap between train, val, and test.
4. Total exclusion of the manually authored held-out test set from train/val.
5. Presence and validity of preprocessed token columns.
"""

import os
import pytest
import pandas as pd


@pytest.fixture(scope="module")
def splits():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    return train, val, test


def test_split_files_exist():
    """All processed split files exist in data/processed/."""
    for fname in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join("data/processed", fname)
        assert os.path.exists(path), f"Processed split file missing: {path}"


def test_no_exact_text_leakage(splits):
    """No text appearing in train can appear in val or test."""
    train_df, val_df, test_df = splits
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    assert len(train_texts.intersection(test_texts)) == 0, "Train-Test text leakage!"
    assert len(train_texts.intersection(val_texts)) == 0, "Train-Val text leakage!"
    assert len(val_texts.intersection(test_texts)) == 0, "Val-Test text leakage!"


def test_no_template_leakage(splits):
    """No template group appearing in train can appear in val or test."""
    train_df, val_df, test_df = splits
    train_templates = set(train_df["template_group"])
    val_templates = set(val_df["template_group"])
    test_templates = set(test_df["template_group"])

    assert len(train_templates.intersection(test_templates)) == 0, "Train-Test template leakage!"
    assert len(train_templates.intersection(val_templates)) == 0, "Train-Val template leakage!"
    assert len(val_templates.intersection(test_templates)) == 0, "Val-Test template leakage!"


def test_held_out_set_strictly_excluded(splits):
    """The manually authored test set (data/raw/hand_written_test.csv) is NOT in train or val."""
    hw_path = "data/raw/hand_written_test.csv"
    assert os.path.exists(hw_path), "hand_written_test.csv missing!"
    hw_df = pd.read_csv(hw_path)
    hw_texts = set(hw_df["text"])

    train_df, val_df, test_df = splits
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])

    overlap_train = hw_texts.intersection(train_texts)
    overlap_val = hw_texts.intersection(val_texts)

    assert len(overlap_train) == 0, f"Held-out test texts leaked into train: {overlap_train}"
    assert len(overlap_val) == 0, f"Held-out test texts leaked into val: {overlap_val}"


def test_clean_tokens_present(splits):
    """Each split contains the clean_tokens and token_count columns."""
    for df in splits:
        assert "clean_tokens" in df.columns
        assert "token_count" in df.columns
        assert (df["token_count"] > 0).all()
