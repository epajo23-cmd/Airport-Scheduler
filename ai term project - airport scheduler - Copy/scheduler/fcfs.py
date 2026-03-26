from __future__ import annotations
import pandas as pd


def fcfs_order(df: pd.DataFrame) -> pd.DataFrame:
    # First-Come-First-Served = sort by scheduled departure time
    return df.sort_values("DEP_MINUTES", ascending=True)
