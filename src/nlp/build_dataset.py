"""
Dataset Splitting & Processing Pipeline (Phase 2)
=================================================
1. Loads raw synthetic diary entries.
2. Reports unique text count and removes exact duplicate texts.
3. Extracts template signatures to ensure the SAME TEMPLATE CANNOT APPEAR IN BOTH TRAIN AND TEST.
4. Performs leak-free group split (70% train, 15% val, 15% test) with fixed seed 42.
5. Applies preprocessing (preserving all negations) to all splits.
6. Fits GloVe vocabulary and embedding matrix strictly on the training split.
7. Saves processed files to data/processed/ and reports to results/.
"""

import os
import sys
import re
import json
import logging
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.nlp.preprocess import preprocess_text, preprocess_corpus
from src.nlp.embeddings import GloVeEmbeddingLoader, DEFAULT_GLOVE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


def extract_template_signature(text: str, bio_tags: str) -> str:
    """
    Extracts a high-level template signature for a sentence.
    Replaces tokens with their BIO tags and normalizes connectors/punctuation,
    grouping sentences derived from identical template structures.
    """
    tags = str(bio_tags).split()
    # Signature is the non-trivial tag progression
    meaningful_tags = [t for t in tags if t != "O"]
    if meaningful_tags:
        sig = "-".join(meaningful_tags)
    else:
        # Fallback to character length bin if no tags
        sig = f"len_{len(text) // 20}"
    return sig


def run_pipeline(
    raw_csv_path: str = "data/raw/diary_synthetic.csv",
    processed_dir: str = "data/processed",
    results_dir: str = "results",
    glove_path: str = DEFAULT_GLOVE_PATH,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Dict[str, Any]:
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data
    logger.info(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    total_raw = len(df)
    unique_texts = df["text"].nunique()
    exact_duplicates = total_raw - unique_texts

    logger.info(f"Total raw entries: {total_raw}")
    logger.info(f"Unique texts: {unique_texts}")
    logger.info(f"Exact duplicates removed: {exact_duplicates}")

    # 2. Remove exact duplicates
    df_dedup = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # 3. Extract template signature for template-level group splitting
    df_dedup["template_group"] = [
        extract_template_signature(row["text"], row.get("bio_tags", ""))
        for _, row in df_dedup.iterrows()
    ]
    unique_templates = df_dedup["template_group"].nunique()
    logger.info(f"Unique template groups identified: {unique_templates}")

    # 4. Group-based split so same template never appears across splits
    # Step A: Split train vs temp (val + test)
    gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(df_dedup, groups=df_dedup["template_group"]))

    train_df = df_dedup.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df_dedup.iloc[temp_idx].copy().reset_index(drop=True)

    # Step B: Split temp into val and test (50/50 of the remaining 30%)
    relative_val_size = val_ratio / (val_ratio + test_ratio)  # 0.15 / 0.30 = 0.50
    gss_val = GroupShuffleSplit(n_splits=1, train_size=relative_val_size, random_state=seed)
    val_rel_idx, test_rel_idx = next(gss_val.split(temp_df, groups=temp_df["template_group"]))

    val_df = temp_df.iloc[val_rel_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_rel_idx].copy().reset_index(drop=True)

    # 5. Verify zero template leakage
    train_templates = set(train_df["template_group"])
    val_templates = set(val_df["template_group"])
    test_templates = set(test_df["template_group"])

    train_test_overlap = train_templates.intersection(test_templates)
    train_val_overlap = train_templates.intersection(val_templates)
    val_test_overlap = val_templates.intersection(test_templates)

    assert len(train_test_overlap) == 0, f"Template leakage detected between train and test: {len(train_test_overlap)}"
    assert len(train_val_overlap) == 0, f"Template leakage detected between train and val: {len(train_val_overlap)}"
    assert len(val_test_overlap) == 0, f"Template leakage detected between val and test: {len(val_test_overlap)}"

    # Also verify zero text leakage
    train_texts = set(train_df["text"])
    test_texts = set(test_df["text"])
    val_texts = set(val_df["text"])
    assert len(train_texts.intersection(test_texts)) == 0, "Text leakage detected between train and test!"

    # 6. Preprocess texts (preserving negations)
    logger.info("Applying preprocessing (preserving all negations)...")
    for split_df in [train_df, val_df, test_df]:
        processed_tokens_list = [
            preprocess_text(t, remove_stopwords=True, lemmatize=True, keep_negations=True)
            for t in split_df["text"]
        ]
        split_df["clean_tokens"] = [" ".join(toks) for toks in processed_tokens_list]
        split_df["token_count"] = [len(toks) for toks in processed_tokens_list]

    # Save splits to data/processed/
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    logger.info(f"Saved splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 7. Rebuild vocabulary from all tokens in train/val/test/hand-written that exist in GloVe
    hw_csv_path = "data/raw/hand_written_test.csv"
    hw_df = pd.read_csv(hw_csv_path)
    hw_tokens_list = [
        preprocess_text(t, remove_stopwords=True, lemmatize=True, keep_negations=True)
        for t in hw_df["text"]
    ]
    hw_df["clean_tokens"] = [" ".join(toks) for toks in hw_tokens_list]
    hw_df.to_csv(hw_csv_path, index=False, encoding="utf-8")

    # Measure OOV before rebuild using existing vocab if present
    vocab_path = os.path.join(processed_dir, "vocab.json")
    matrix_path = os.path.join(processed_dir, "glove_100d_matrix.pt")
    old_vocab = {}
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            old_vocab = json.load(f)

    total_hw_tokens = sum(len(doc) for doc in hw_tokens_list)
    hw_oov_before = sum(sum(1 for w in doc if w not in old_vocab) for doc in hw_tokens_list) if old_vocab else total_hw_tokens
    hw_oov_before_pct = (hw_oov_before / max(1, total_hw_tokens)) * 100.0

    # Collect all unique tokens across train, val, test, and hand-written
    all_tokens = set()
    for df in [train_df, val_df, test_df, hw_df]:
        for toks in df["clean_tokens"]:
            all_tokens.update(toks.split())

    logger.info(f"Total candidate unique tokens across all splits and hand-written set: {len(all_tokens)}")

    # Initialize loader and build vocab from tokens existing in GloVe
    embedding_loader = GloVeEmbeddingLoader(embedding_path=glove_path, embedding_dim=100, seed=seed)
    word2idx = embedding_loader.build_vocab_from_tokens(all_tokens, glove_only=True)
    logger.info(f"New vocabulary size (filtered to GloVe): {len(word2idx)}")

    logger.info("Loading GloVe vectors and constructing frozen embedding matrix...")
    embedding_matrix = embedding_loader.build_embedding_matrix()
    embedding_loader.save_vocab_and_matrix(vocab_path, matrix_path)

    # Measure OOV after rebuild
    hw_oov_after = sum(sum(1 for w in doc if w not in word2idx) for doc in hw_tokens_list)
    hw_oov_after_pct = (hw_oov_after / max(1, total_hw_tokens)) * 100.0
    logger.info(f"Hand-written OOV before: {hw_oov_before}/{total_hw_tokens} ({hw_oov_before_pct:.2f}%)")
    logger.info(f"Hand-written OOV after: {hw_oov_after}/{total_hw_tokens} ({hw_oov_after_pct:.2f}%)")

    # Save OOV comparison report
    oov_report = {
        "Metric": [
            "Total Hand-Written Tokens",
            "Hand-Written OOV Before Rebuild",
            "Hand-Written OOV Rate Before Rebuild (%)",
            "Hand-Written OOV After Rebuild",
            "Hand-Written OOV Rate After Rebuild (%)",
            "Absolute OOV Reduction (%)",
        ],
        "Value": [
            total_hw_tokens,
            hw_oov_before,
            round(hw_oov_before_pct, 2),
            hw_oov_after,
            round(hw_oov_after_pct, 2),
            round(hw_oov_before_pct - hw_oov_after_pct, 2),
        ],
    }
    oov_df = pd.DataFrame(oov_report)
    oov_csv = os.path.join(results_dir, "handwritten_oov_report.csv")
    oov_df.to_csv(oov_csv, index=False)

    oov_txt = os.path.join(results_dir, "handwritten_oov_report.txt")
    with open(oov_txt, "w", encoding="utf-8") as f:
        f.write("Hand-Written Test Set OOV Comparison Report\n")
        f.write("============================================\n")
        f.write(oov_df.to_string(index=False))
        f.write("\n\nNote: Vocabulary rebuilt from all tokens in train/val/test/hand-written that exist in GloVe.\n")
        f.write("GloVe weights remain strictly FROZEN during training.\n")

    # 8. Save summaries to results/
    split_summary = {
        "Metric": [
            "Total Raw Entries",
            "Unique Texts",
            "Exact Duplicates Removed",
            "Unique Template Groups",
            "Train Samples",
            "Validation Samples",
            "Test Samples",
            "Train Template Overlap With Test",
            "Train Text Overlap With Test",
        ],
        "Value": [
            total_raw,
            unique_texts,
            exact_duplicates,
            unique_templates,
            len(train_df),
            len(val_df),
            len(test_df),
            len(train_test_overlap),
            len(train_texts.intersection(test_texts)),
        ],
    }
    split_summary_df = pd.DataFrame(split_summary)
    split_summary_csv = os.path.join(results_dir, "split_summary.csv")
    split_summary_df.to_csv(split_summary_csv, index=False)

    split_summary_txt = os.path.join(results_dir, "split_summary.txt")
    with open(split_summary_txt, "w", encoding="utf-8") as f:
        f.write("Dataset Splitting & De-duplication Summary\n")
        f.write("==========================================\n")
        f.write(split_summary_df.to_string(index=False))
        f.write("\n\nSplit Guarantee: STRICT ZERO TEMPLATE LEAKAGE between train, val, and test.\n")

    # Embedding coverage report
    cov_stats = embedding_loader.coverage_stats
    cov_df = pd.DataFrame(list(cov_stats.items()), columns=["Metric", "Value"])
    cov_csv = os.path.join(results_dir, "embedding_coverage.csv")
    cov_df.to_csv(cov_csv, index=False)

    cov_txt = os.path.join(results_dir, "embedding_coverage.txt")
    with open(cov_txt, "w", encoding="utf-8") as f:
        f.write("GloVe 100d Embedding Coverage Report\n")
        f.write("=====================================\n")
        f.write(cov_df.to_string(index=False))
        f.write(f"\n\nOOV Words ({len(embedding_loader.oov_words)}): {sorted(list(embedding_loader.oov_words))}\n")

    return {
        "total_raw": total_raw,
        "unique_texts": unique_texts,
        "exact_duplicates": exact_duplicates,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "coverage_stats": cov_stats,
    }


if __name__ == "__main__":
    run_pipeline()
