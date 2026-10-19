"""
Text Preprocessing Module
=========================
Handles tokenisation, lowercasing, contraction expansion, lemmatisation,
and stop-word filtering for hand-rehabilitation daily diaries.

CRITICAL RULE (from AGENTS.md):
Preprocessing MUST KEEP negations ("no", "not", "never", "without", "zero", etc.).
Negation words must NEVER be removed as stop-words.
"""

import re
from typing import List, Set, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources
for resource in ["wordnet", "punkt", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource != "punkt" else "tokenizers/punkt")
    except LookupError:
        nltk.download(resource, quiet=True)

# Negation words that MUST be preserved under all circumstances
NEGATION_WORDS: Set[str] = {
    "no",
    "not",
    "never",
    "without",
    "none",
    "neither",
    "nor",
    "barely",
    "hardly",
    "zero",
    "cannot",
    "cant",
    "couldnt",
    "wont",
    "dont",
    "didnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "free",
    "absent",
}

# Standard English contractions
CONTRACTIONS = {
    r"\bcan'?t\b": "cannot",
    r"\bcouldn'?t\b": "could not",
    r"\bdon'?t\b": "do not",
    r"\bdidn'?t\b": "did not",
    r"\bwon'?t\b": "will not",
    r"\bisn'?t\b": "is not",
    r"\baren'?t\b": "are not",
    r"\bwasn'?t\b": "was not",
    r"\bweren'?t\b": "were not",
    r"\bhasn'?t\b": "has not",
    r"\bhaven'?t\b": "have not",
    r"\bhadn'?t\b": "had not",
    r"\bw/o\b": "without",
    r"\b2day\b": "today",
    r"\bmornin\b": "morning",
    r"\bfingrs\b": "fingers",
    r"\bexcercises\b": "exercises",
    r"\balot\b": "a lot",
    r"\btho\b": "though",
}

_lemmatizer = WordNetLemmatizer()

try:
    _raw_stopwords = set(stopwords.words("english"))
except Exception:
    _raw_stopwords = set()

# Safe stopword set: standard stopwords MINUS all negation words
SAFE_STOPWORDS: Set[str] = _raw_stopwords - NEGATION_WORDS


def expand_contractions(text: str) -> str:
    """Expands contractions and informal abbreviations."""
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenizes text into words and punctuation."""
    return re.findall(r"\b\w+\b|[^\w\s]", text)


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    keep_negations: bool = True,
) -> List[str]:
    """
    Preprocesses raw diary text:
    1. Expands contractions (e.g. 'can't' -> 'cannot')
    2. Lowercases
    3. Tokenizes into words
    4. Filters non-negation stop-words (preserving all negations)
    5. Lemmatizes tokens
    
    Returns list of processed tokens.
    """
    if not text or not isinstance(text, str):
        return []

    # 1. Contraction expansion
    expanded = expand_contractions(text)

    # 2. Lowercase
    lowered = expanded.lower()

    # 3. Extract alphanumeric tokens
    raw_tokens = re.findall(r"\b[a-z0-9]+\b", lowered)

    processed_tokens: List[str] = []
    for token in raw_tokens:
        # Check stopword removal
        if remove_stopwords:
            if keep_negations and token in NEGATION_WORDS:
                pass  # Explicitly preserve negation token
            elif token in SAFE_STOPWORDS:
                continue

        # Lemmatize
        if lemmatize:
            lemma = _lemmatizer.lemmatize(token)
            # Try verb lemmatization if unchanged
            if lemma == token:
                lemma = _lemmatizer.lemmatize(token, pos="v")
            processed_tokens.append(lemma)
        else:
            processed_tokens.append(token)

    return processed_tokens


def preprocess_corpus(
    texts: List[str],
    remove_stopwords: bool = True,
    lemmatize: bool = True,
) -> List[List[str]]:
    """Applies preprocessing to an iterable of texts."""
    return [
        preprocess_text(t, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
        for t in texts
    ]
