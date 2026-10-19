"""
Classical NLP Baseline: TF-IDF + Logistic Regression (Phase 3)
==============================================================
Multi-head classical baseline using TF-IDF n-gram vectorization and
Logistic Regression classifiers for pain, stiffness, mobility, and grip severity (0-3).
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


SYMPTOMS = ["pain", "stiffness", "mobility", "grip"]


class TfidfLogRegBaseline:
    def __init__(self, random_state: int = 42, max_features: int = 1000):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.models: Dict[str, LogisticRegression] = {}
        for sym in SYMPTOMS:
            self.models[sym] = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
                class_weight=None,
            )

    def fit(self, texts: List[str], targets: Dict[str, np.ndarray]) -> "TfidfLogRegBaseline":
        """Fits TF-IDF vectorizer and the 4 Logistic Regression heads."""
        X = self.vectorizer.fit_transform(texts)
        for sym in SYMPTOMS:
            self.models[sym].fit(X, targets[sym])
        return self

    def predict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Predicts severity (0-3) for each symptom head."""
        X = self.vectorizer.transform(texts)
        preds = {}
        for sym in SYMPTOMS:
            preds[sym] = self.models[sym].predict(X)
        return preds

    def evaluate(
        self,
        texts: List[str],
        targets: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates predictions against ground truth targets for all 4 symptoms.
        Returns accuracy, macro precision, macro recall, and macro F1 per symptom.
        """
        preds = self.predict(texts)
        metrics: Dict[str, Dict[str, float]] = {}

        for sym in SYMPTOMS:
            y_true = targets[sym]
            y_pred = preds[sym]
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
            metrics[sym] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }

        # Compute overall average macro F1 across the 4 symptoms
        mean_macro_f1 = np.mean([metrics[sym]["f1"] for sym in SYMPTOMS])
        mean_acc = np.mean([metrics[sym]["accuracy"] for sym in SYMPTOMS])
        metrics["overall"] = {
            "mean_macro_f1": float(mean_macro_f1),
            "mean_accuracy": float(mean_acc),
        }
        return metrics
