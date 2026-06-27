from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml import clean_text


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR / "data" / "sample_emails.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "phishing_model.joblib"


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected_columns = {"text", "label"}

    if not expected_columns.issubset(data.columns):
        raise ValueError("Dataset must include 'text' and 'label' columns.")

    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip()

    valid_labels = {"phishing", "legitimate"}
    unknown_labels = set(data["label"]) - valid_labels
    if unknown_labels:
        raise ValueError(f"Unsupported labels found: {sorted(unknown_labels)}")

    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train(data_path: Path, model_path: Path) -> None:
    dataset = load_dataset(data_path)
    train_text, test_text, train_labels, test_labels = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=0.25,
        random_state=42,
        stratify=dataset["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(train_text, train_labels)

    predictions = pipeline.predict(test_text)
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(test_labels, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, predictions, labels=["phishing", "legitimate"]))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the phishing email classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Path to a labeled CSV dataset.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Output path for the trained model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model)
