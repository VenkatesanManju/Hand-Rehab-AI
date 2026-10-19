"""
Unit Tests for Preprocessing Module (Phase 2)
=============================================
Tests lowercasing, tokenisation, lemmatisation, contraction expansion,
and strictly checks that negations are KEPT.
"""

import pytest
from src.nlp.preprocess import (
    preprocess_text,
    expand_contractions,
    preprocess_corpus,
    NEGATION_WORDS,
)


def test_no_pain_but_stiff():
    """Explicit MASTER_PLAN requirement: Unit test for 'no pain but stiff'."""
    tokens = preprocess_text("no pain but stiff")
    assert "no" in tokens, "'no' was removed from 'no pain but stiff'!"
    assert "pain" in tokens, "'pain' was removed!"
    assert "stiff" in tokens, "'stiff' was removed!"
    # 'but' is a standard stopword and safely removed
    assert "but" not in tokens


def test_all_negations_preserved():
    """Every negation word in NEGATION_WORDS must be retained through preprocessing."""
    for neg in ["no", "not", "never", "without", "zero", "cannot", "free", "barely"]:
        sentence = f"{neg} symptom detected"
        tokens = preprocess_text(sentence)
        assert neg in tokens, f"Negation '{neg}' was erroneously removed by stopword filter!"


def test_contraction_expansion():
    """Contractions like can't, couldn't, didn't expand to preserve negation words."""
    text = "can't hold a mug, didn't feel stiffness"
    tokens = preprocess_text(text)
    # can't -> cannot, didn't -> did not
    assert "cannot" in tokens or "not" in tokens
    assert "not" in tokens


def test_lemmatization():
    """Inflected word forms are reduced to lemma."""
    text = "fingers were aching and hurting during exercises"
    tokens = preprocess_text(text)
    # aching -> ache, hurting -> hurt, fingers -> finger, exercises -> exercise
    assert "finger" in tokens
    assert "ache" in tokens
    assert "hurt" in tokens
    assert "exercise" in tokens


def test_case_insensitivity():
    """Uppercase, titlecase, lowercase produce identical output."""
    t1 = preprocess_text("NO PAIN BUT VERY STIFF")
    t2 = preprocess_text("No Pain But Very Stiff")
    t3 = preprocess_text("no pain but very stiff")
    assert t1 == t2 == t3
    assert "no" in t1


def test_empty_and_special_characters():
    """Empty strings and punctuation handle gracefully."""
    assert preprocess_text("") == []
    assert preprocess_text(None) == []
    assert preprocess_text("!@#$%^&*()_+") == []


def test_informal_typos_normalized():
    """Informal abbreviations like w/o, 2day, mornin are normalized."""
    text = "w/o pain 2day mornin"
    tokens = preprocess_text(text)
    assert "without" in tokens
    assert "pain" in tokens
    assert "today" in tokens
    assert "morning" in tokens


def test_preprocess_corpus():
    """Batch preprocessing works consistently."""
    corpus = [
        "No pain today.",
        "Sharp pain when I tried to close my fist.",
        "Thumb barely moves.",
    ]
    processed = preprocess_corpus(corpus)
    assert len(processed) == 3
    assert "no" in processed[0]
    assert "pain" in processed[1]
    assert "barely" in processed[2]
