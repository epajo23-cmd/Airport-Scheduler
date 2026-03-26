from __future__ import annotations

import os
import sys
import joblib
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


from utils.preprocess import load_and_prepare, split_xy


MODEL_PATH = os.path.join("model", "model.pkl")


def build_pipeline() -> Pipeline:
    """
    Simple + strong baseline:
    - OneHotEncode categorical features
    - Pass numeric features through
    - Logistic Regression classifier
    """
    categorical = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    numeric = ["MONTH", "DAY_OF_WEEK", "DEP_MINUTES", "DISTANCE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )


    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )


def main():
    df = load_and_prepare("data/flights.csv", sample_n=200_000)
    X, y = split_xy(df, task="classification")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report (includes F1) ===")
    print(classification_report(y_test, y_pred, digits=4))
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
