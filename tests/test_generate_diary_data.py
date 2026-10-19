"""
Tests for Diary Dataset Generator (Phase 1)
===========================================
Validates row counts, label schema, class balance, BIO tag token alignment,
and preservation of raw seed files.
"""

import os
import pytest
import pandas as pd
from src.nlp.generate_diary_data import (
    generate_synthetic_dataset,
    validate_label_distribution,
    generate_hand_written_test_set,
    SEED,
)


@pytest.fixture(scope="module")
def synthetic_df():
    csv_path = "data/raw/diary_synthetic.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_synthetic_dataset(n_samples=2500, seed=SEED)


def test_reproducibility():
    """Generating dataset with same seed produces identical results."""
    df1 = generate_synthetic_dataset(n_samples=50, seed=42)
    df2 = generate_synthetic_dataset(n_samples=50, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_row_count(synthetic_df):
    """Synthetic dataset must contain at least 2,000 entries."""
    assert len(synthetic_df) >= 2000, f"Expected >= 2000 rows, got {len(synthetic_df)}"


def test_required_columns(synthetic_df):
    """Required columns per labelling schema and master plan must be present."""
    required = ["text", "pain", "stiffness", "mobility", "grip", "source", "bio_tags", "tokens"]
    for col in required:
        assert col in synthetic_df.columns, f"Missing required column: {col}"


def test_severity_ranges(synthetic_df):
    """All severity scores must be integers between 0 and 3."""
    symptoms = ["pain", "stiffness", "mobility", "grip"]
    for sym in symptoms:
        unique_vals = set(synthetic_df[sym].unique())
        assert unique_vals.issubset({0, 1, 2, 3}), f"Invalid values in {sym}: {unique_vals}"


def test_label_distribution_balance(synthetic_df):
    """Every class (0-3) must have >= 10% representation per symptom."""
    dist = validate_label_distribution(synthetic_df, min_class_pct=10.0)
    for sym, class_pcts in dist.items():
        for c, pct in class_pcts.items():
            assert pct >= 10.0, f"Class {c} for {sym} only has {pct}% (< 10%)"


def test_bio_tag_alignment(synthetic_df):
    """Token count and BIO tag count must align 1-to-1 for every single entry."""
    for idx, row in synthetic_df.iterrows():
        toks = str(row["tokens"]).split()
        tags = str(row["bio_tags"]).split()
        assert len(toks) == len(tags), (
            f"Row {idx} tag mismatch: {len(toks)} tokens vs {len(tags)} tags. "
            f"Tokens: {toks[:5]}... Tags: {tags[:5]}..."
        )


def test_synthetic_source_label(synthetic_df):
    """Synthetic data must be explicitly labeled as synthetic in the source column."""
    sources = set(synthetic_df["source"].unique())
    for s in sources:
        assert "synthetic" in s.lower(), f"Source must indicate synthetic, got: {s}"


def test_negations_present(synthetic_df):
    """Dataset must contain explicit negation entries ('no', 'not', 'zero', 'without')."""
    negation_words = {"no", "not", "zero", "without", "free"}
    has_negation = False
    for text in synthetic_df["text"]:
        words = set(text.lower().split())
        if words.intersection(negation_words):
            has_negation = True
            break
    assert has_negation, "Dataset does not contain negation phrases"


def test_raw_seed_files_preserved():
    """Seed examples file must exist in data/raw/ and not be overwritten/deleted."""
    seed_path = "data/raw/seed_examples.jsonl"
    assert os.path.exists(seed_path), "seed_examples.jsonl was deleted!"
    with open(seed_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) >= 8, f"seed_examples.jsonl corrupted: found {len(lines)} lines"


def test_hand_written_test_set():
    hw_df = generate_hand_written_test_set()
    assert len(hw_df) >= 50, f"Expected >= 50 held-out test rows, got {len(hw_df)}"
    assert "bio_tags" in hw_df.columns
    valid_sources = {"hand_written", "agent_written"}
    assert set(hw_df["source"].unique()).issubset(valid_sources)
    assert (hw_df["source"] == "hand_written").sum() >= 8
    assert (hw_df["source"] == "agent_written").sum() >= 40
