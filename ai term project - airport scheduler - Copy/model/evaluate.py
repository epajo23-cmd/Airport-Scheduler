from __future__ import annotations

import os
import sys
import joblib
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.preprocess import load_and_prepare, split_xy

MODEL_PATH = os.path.join("model", "model.pkl")


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """
    Minimal matplotlib confusion matrix plot (no fancy styling).
    """
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Not Delayed", "Delayed (15+)"])
    plt.yticks([0, 1], ["Not Delayed", "Delayed (15+)"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    return fig


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python model/train.py first."
        )
    model = joblib.load(MODEL_PATH)
    df = load_and_prepare("data/flights.csv", sample_n=200_000)
    X, y = split_xy(df, task="classification")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print("\n================ EVALUATION ================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("===========================================\n")

    print("=== Confusion Matrix ===")
    print(cm)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    fig = plot_confusion_matrix(cm, title="Flight Delay Classifier (15+ min)")
    plt.show()


if __name__ == "__main__":
    main()
