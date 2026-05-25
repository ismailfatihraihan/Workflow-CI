"""MLflow Project training entry point for GitHub Actions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


TARGET_COLUMN = "diagnosis"


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_frame = pd.read_csv(data_dir / "train.csv")
    test_frame = pd.read_csv(data_dir / "test.csv")
    return (
        train_frame.drop(columns=[TARGET_COLUMN]),
        test_frame.drop(columns=[TARGET_COLUMN]),
        train_frame[TARGET_COLUMN],
        test_frame[TARGET_COLUMN],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("breast_cancer_preprocessing"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    x_train, x_test, y_train, y_test = load_dataset(args.data_dir)

    with mlflow.start_run(run_name="ci-random-forest") as run:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions),
                "test_recall": recall_score(y_test, predictions),
                "test_f1": f1_score(y_test, predictions),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        Path("latest_run_id.txt").write_text(run.info.run_id, encoding="utf-8")


if __name__ == "__main__":
    main()
