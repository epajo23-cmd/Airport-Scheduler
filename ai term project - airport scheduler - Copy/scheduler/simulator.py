from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class SimMetrics:
    avg_total_delay: float
    max_total_delay: float
    avg_waiting_time: float
    avg_idle_time: float
    flights: int


def simulate_queue(
    queue_df: pd.DataFrame,
    order_fn: Callable[[pd.DataFrame], pd.DataFrame],
    separation_minutes: float = 2.0,
    # Risk penalty settings (used to create realistic “cascades”)
    risk_threshold: float = 0.60,
    risk_penalty_minutes: float = 3.0,
) -> tuple[pd.DataFrame, SimMetrics]:
    """
    Runway scheduling simulation with two realistic ideas:

    (1) READY_TIME (a flight might not be ready at its scheduled time)
        READY_TIME = DEP_MINUTES + max(DEPARTURE_DELAY, 0)

    (2) Risk-based separation penalty
        If a flight is high-risk (pred_delay_risk >= risk_threshold),
        we add extra separation AFTER it (like disruptions / extra spacing).

    Metrics:
      waiting_time = assigned_time - DEP_MINUTES
      idle_time    = max(0, READY_TIME - (prev_assigned + prev_sep))
      total_delay_proxy = DEPARTURE_DELAY + waiting_time
    """

    required = ["DEP_MINUTES", "DEPARTURE_DELAY"]
    missing = [c for c in required if c not in queue_df.columns]
    if missing:
        raise ValueError(f"queue_df missing required columns: {missing}")

    df = queue_df.copy()

    # --- READY_TIME approximation ---
    df["READY_TIME"] = df["DEP_MINUTES"] + df["DEPARTURE_DELAY"].clip(lower=0)

    # Order the queue (FCFS or intelligent)
    ordered = order_fn(df).reset_index(drop=True)

    assigned_times = []
    idle_times = []

    # --- Dynamic separation based on risk of the PREVIOUS scheduled flight ---
    RISK_COL = "pred_delay_risk"

    current_time = None
    prev_sep = float(separation_minutes)

    for i, row in ordered.iterrows():
        dep = float(row["DEP_MINUTES"])
        ready = float(row["READY_TIME"])

        if current_time is None:
            current_time = dep

        earliest_slot = current_time if i == 0 else current_time + prev_sep

        # If chosen flight isn't ready, runway must wait idle
        idle = max(0.0, ready - earliest_slot)

        # Actual assigned departure time
        assigned = max(ready, earliest_slot)

        assigned_times.append(assigned)
        idle_times.append(idle)

        # Compute separation for NEXT flight based on THIS flight’s risk
        risk = float(row.get(RISK_COL, 0.0))
        penalty = float(risk_penalty_minutes) if risk >= float(risk_threshold) else 0.0
        prev_sep = float(separation_minutes) + penalty

        current_time = assigned

    ordered["assigned_time"] = assigned_times
    ordered["idle_time"] = idle_times

    # Waiting vs schedule
    ordered["waiting_time"] = (ordered["assigned_time"] - ordered["DEP_MINUTES"]).clip(lower=0.0)
    ordered["total_delay_proxy"] = ordered["DEPARTURE_DELAY"] + ordered["waiting_time"]

    metrics = SimMetrics(
        avg_total_delay=float(ordered["total_delay_proxy"].mean()),
        max_total_delay=float(ordered["total_delay_proxy"].max()),
        avg_waiting_time=float(ordered["waiting_time"].mean()),
        avg_idle_time=float(ordered["idle_time"].mean()),
        flights=int(len(ordered)),
    )

    return ordered, metrics


def metrics_to_dict(m: SimMetrics) -> Dict[str, float]:
    return {
        "flights": m.flights,
        "avg_total_delay": m.avg_total_delay,
        "max_total_delay": m.max_total_delay,
        "avg_waiting_time": m.avg_waiting_time,
        "avg_idle_time": m.avg_idle_time,
    }
