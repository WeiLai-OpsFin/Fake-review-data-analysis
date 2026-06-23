"""Reusable fake-review text classification pipeline.

This script is intentionally compact and portfolio-friendly. It accepts any CSV
with a text column and a label column, cleans the text, builds TF-IDF features,
and evaluates a logistic-regression classifier.
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_text(text: object) -> str:
    """Normalize review text while preserving enough signal for classification."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline(max_features: int = 20_000) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                ),
            ),
            (
                "model",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file containing review text and labels")
    parser.add_argument("--text-col", default="text_", help="Name of the review text column")
    parser.add_argument("--label-col", default="label", help="Name of the label column")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[[args.text_col, args.label_col]].dropna()
    if df[args.label_col].nunique() < 2:
        raise ValueError("The label column must contain at least two classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        df[args.text_col],
        df[args.label_col],
        test_size=args.test_size,
        random_state=42,
        stratify=df[args.label_col],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_test, pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(outdir / "classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(outdir / "confusion_matrix.csv", index=False)

    print(classification_report(y_test, pred))


if __name__ == "__main__":
    main()
