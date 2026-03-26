from __future__ import annotations

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("model", "model.pkl")


def intelligent_order(
    queue_df: pd.DataFrame,
    separation_minutes: float = 2.0,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """
    Time-aware Intelligent scheduling policy:

    - Predict delay risk for each flight using the trained classifier.
    - Build the runway sequence step-by-step.
    - At each step, choose the HIGHEST risk flight among flights whose DEP_MINUTES <= current_time
      (i.e., flights that are "available" now).
    - If none are available yet, jump time forward to the next earliest DEP_MINUTES.

    This prevents the unrealistic behavior where a late flight is scheduled first,
    causing huge waiting times for earlier flights.
    """

    required = [
        "MONTH",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "DEP_MINUTES",
        "DISTANCE",
    ]
    missing = [c for c in required if c not in queue_df.columns]
    if missing:
        raise ValueError(f"queue_df is missing required columns: {missing}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first: python model/train.py"
        )

    model = joblib.load(model_path)

    df = queue_df.copy().reset_index(drop=True)
    X = df[required].copy()

    # Probability of class 1: Delayed (15+)
    df["pred_delay_risk"] = model.predict_proba(X)[:, 1]

    # Time-aware selection
    remaining = df.copy()
    chosen_rows = []

    # Start runway clock at earliest scheduled departure in the queue
    current_time = float(remaining["DEP_MINUTES"].min())

    while len(remaining) > 0:
        # Flights "available" by current_time
        available = remaining[remaining["DEP_MINUTES"] <= current_time]

        if len(available) == 0:
            # No flights available yet: jump time to next soonest flight
            current_time = float(remaining["DEP_MINUTES"].min())
            available = remaining[remaining["DEP_MINUTES"] <= current_time]

        # Choose highest predicted risk among available
        pick_idx = available["pred_delay_risk"].idxmin()
        picked = remaining.loc[pick_idx]
        chosen_rows.append(picked)

        # Remove picked flight
        remaining = remaining.drop(index=pick_idx)

        # Move runway time forward by separation
        current_time = max(current_time, float(picked["DEP_MINUTES"])) + float(separation_minutes)

    ordered = pd.DataFrame(chosen_rows).reset_index(drop=True)
    return ordered
