"""
Model training and evaluation module.

Loads dataset/processed/documents.csv, preprocesses text, trains a
TF-IDF + classifier pipeline, evaluates it, and saves the trained model
components to models/.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "dataset" / "processed" / "documents.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

STOPWORDS = set(
    """
a an the and or but if then else when at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there all any both each few more most other some such no nor not only own same so than too very s t can will just don should now
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing
""".split()
)


def preprocess_text(text: str) -> str:
    """Lowercase, remove non-alphanumeric, drop stopwords, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the dataset CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"filename", "category", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["category"].astype(str).str.strip() != ""]

    df["text_clean"] = df["text"].apply(preprocess_text)
    df = df[df["text_clean"].str.strip() != ""]

    return df.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    """Build a TF-IDF + LinearSVC classification pipeline."""
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    clf = LinearSVC(C=1.0, dual="auto", random_state=42)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Train the model and return evaluation metrics.
    Also saves model artifacts and a report.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(CSV_PATH)
    print(f"Loaded {len(df)} documents")
    print(f"Categories: {df['category'].value_counts().to_dict()}")

    le = LabelEncoder()
    y = le.fit_transform(df["category"])
    X = df["text_clean"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    print("Training model...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "train_time_seconds": round(train_time, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "categories": list(le.classes_),
        "category_counts": df["category"].value_counts().to_dict(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"\nClassification Report:\n{report}")

    joblib.dump(pipeline, MODELS_DIR / "classifier_pipeline.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    with open(REPORTS_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(REPORTS_DIR / "evaluation.txt", "w", encoding="utf-8") as f:
        f.write("Document Classification - Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {CSV_PATH.name}\n")
        f.write(f"Total documents: {len(df)}\n")
        f.write(f"Train/Test split: {len(X_train)}/{len(X_test)} "
                f"({100-test_size*100:.0f}%/{test_size*100:.0f}%)\n")
        f.write(f"Categories: {list(le.classes_)}\n")
        f.write(f"Category counts: {df['category'].value_counts().to_dict()}\n\n")
        f.write(f"Model: TF-IDF (1-2 grams) + LinearSVC\n")
        f.write(f"Training time: {train_time:.2f}s\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 (macro): {f1_macro:.4f}\n")
        f.write(f"F1 (weighted): {f1_weighted:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  Rows=true, Cols=predicted, order={list(le.classes_)}\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\nModel saved to {MODELS_DIR / 'classifier_pipeline.pkl'}")
    print(f"Label encoder saved to {MODELS_DIR / 'label_encoder.pkl'}")
    print(f"Report saved to {REPORTS_DIR / 'evaluation.txt'}")

    return metrics


def predict_text(text: str) -> tuple:
    """
    Load the saved model and predict the category of a single text.
    Returns (category_label, confidence_score).
    """
    pipeline_path = MODELS_DIR / "classifier_pipeline.pkl"
    le_path = MODELS_DIR / "label_encoder.pkl"

    if not pipeline_path.exists() or not le_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    pipeline = joblib.load(pipeline_path)
    le = joblib.load(le_path)

    clean = preprocess_text(text)
    if not clean.strip():
        return "Unknown", 0.0

    pred = pipeline.predict([clean])[0]
    label = le.inverse_transform([pred])[0]

    if hasattr(pipeline.named_steps["clf"], "decision_function"):
        scores = pipeline.decision_function([clean])
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            probs = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
            confidence = float(np.max(probs))
        else:
            confidence = float(abs(scores[0])) / (1 + abs(scores[0]))
    else:
        proba = pipeline.predict_proba([clean])
        confidence = float(np.max(proba))

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train document classification model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_and_evaluate(test_size=args.test_size, random_state=args.random_state)
