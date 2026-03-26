from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, List
FEATURES: List[str] = [
    "MONTH",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "DISTANCE",
]

TARGET_DELAY_MINUTES = "DEPARTURE_DELAY"


def _scheduled_departure_to_minutes(value) -> float:
    """
    Convert SCHEDULED_DEPARTURE like 5, 45, 930, 1545 into minutes-from-midnight.
    Kaggle flights.csv uses "hhmm" format but may appear as int/float/str.
    Returns np.nan if unusable.
    """
    if pd.isna(value):
        return np.nan

    # Convert to int safely (handles "930.0" or "0930")
    try:
        s = str(int(float(value))).zfill(4)  # "930" -> "0930"
    except Exception:
        return np.nan

    hh = int(s[:2])
    mm = int(s[2:])
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return np.nan
    return float(hh * 60 + mm)


def load_and_prepare(
    csv_path: str,
    sample_n: int | None = 200_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads flights.csv, applies minimal cleaning, and prepares columns for ML + scheduling.

    - Filters cancelled/diverted flights
    - Creates:
        DEP_MINUTES = minutes from midnight for scheduled departure time
        DELAYED_15 = classification label (1 if departure_delay > 15 else 0)
    - Keeps only the minimal columns needed
    - Optionally samples to keep training fast (recommended)
    """
    usecols = FEATURES + [
        TARGET_DELAY_MINUTES,
        "CANCELLED",
        "DIVERTED",
    ]

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    # Keep only completed, non-diverted flights
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
    # Force categoricals to string to avoid int/str mix issues in OneHotEncoder
    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        df[col] = df[col].astype(str).str.strip()


    # Convert scheduled departure to minutes
    df["DEP_MINUTES"] = df["SCHEDULED_DEPARTURE"].apply(_scheduled_departure_to_minutes)

    # Remove rows with missing essentials
    df = df.dropna(subset=["DEP_MINUTES", TARGET_DELAY_MINUTES, "DISTANCE", "AIRLINE",
                           "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH", "DAY_OF_WEEK"])

    df["DEP_MINUTES"] = pd.to_numeric(df["DEP_MINUTES"], errors="coerce")
    df["DISTANCE"] = pd.to_numeric(df["DISTANCE"], errors="coerce")
    df[TARGET_DELAY_MINUTES] = pd.to_numeric(df[TARGET_DELAY_MINUTES], errors="coerce")

    df = df.dropna(subset=["DEP_MINUTES", "DISTANCE", TARGET_DELAY_MINUTES])
    df["DELAYED_15"] = (df[TARGET_DELAY_MINUTES] > 15).astype(int)
    keep = [
        "MONTH",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "DEP_MINUTES",
        "DISTANCE",
        TARGET_DELAY_MINUTES,
        "DELAYED_15",
    ]
    df = df[keep].copy()
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).copy()

    return df


def split_xy(
    df: pd.DataFrame,
    task: str = "classification",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns X, y for either:
    - task="classification": y = DELAYED_15
    - task="regression": y = DEPARTURE_DELAY
    """
    X = df[
        [
            "MONTH",
            "DAY_OF_WEEK",
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "DEP_MINUTES",
            "DISTANCE",
        ]
    ].copy()

    if task == "classification":
        y = df["DELAYED_15"].copy()
    elif task == "regression":
        y = df[TARGET_DELAY_MINUTES].copy()
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    return X, y
