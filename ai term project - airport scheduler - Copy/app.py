import os
import joblib
import streamlit as st
import pandas as pd

from utils.preprocess import load_and_prepare 
from scheduler.fcfs import fcfs_order
from scheduler.intelligent import intelligent_order
from scheduler.simulator import simulate_queue, metrics_to_dict

MODEL_PATH = os.path.join("model", "model.pkl")

RISK_FEATURES = [
    "MONTH",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DEP_MINUTES",
    "DISTANCE",
]


def add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first:\n"
            f"  python model/train.py"
        )
    model = joblib.load(MODEL_PATH)
    X = df[RISK_FEATURES].copy()
    out = df.copy()
    out["pred_delay_risk"] = model.predict_proba(X)[:, 1]
    return out


st.set_page_config(page_title="Runway Delay Agent", layout="wide")

st.title("Runway Delay Agent")
st.sidebar.header("Controls")

queue_size = st.sidebar.slider("Queue size (number of flights)", 10, 200, 200, 10)
separation = st.sidebar.slider("Runway separation (minutes)", 0.5, 10.0, 10.0, 0.5)

sample_rows = st.sidebar.slider(
    "Dataset sample used in app (speed vs realism)", 50_000, 400_000, 200_000, 50_000
)

seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Optional filter")
filter_origin = st.sidebar.checkbox("Filter by origin airport", value=False)
origin_airport = st.sidebar.text_input("Origin airport (e.g., JFK, ATL)", value="JFK").strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Risk penalty settings")
risk_threshold = st.sidebar.slider("Risk threshold", 0.3, 0.9, 0.60, 0.05)
risk_penalty = st.sidebar.slider("Extra separation after risky flight (minutes)", 0.0, 10.0, 3.0, 0.5)
if "queue_df" not in st.session_state:
    st.session_state.queue_df = None

colA, colB = st.columns([2, 1])

with colA:
    if st.button("Sample Queue"):
        df = load_and_prepare(
            csv_path=os.path.join("data", "flights.csv"),
            sample_n=int(sample_rows),
            random_state=int(seed),
        )

        if filter_origin and origin_airport:
            df = df[df["ORIGIN_AIRPORT"].astype(str).str.upper() == origin_airport]

        queue_df = df.sample(n=int(queue_size), random_state=int(seed)).reset_index(drop=True)
        queue_df = add_risk_scores(queue_df)

        st.session_state.queue_df = queue_df
        st.success(f"Queue sampled: {len(queue_df)} flights")

with colB:
    st.markdown("### Run schedulers")
    run_fcfs = st.button("Run FCFS")
    run_intel = st.button("Run Intelligent")

st.markdown("")

if st.session_state.queue_df is not None:
    with st.expander("Preview sampled queue (first 20 rows)", expanded=False):
        st.dataframe(st.session_state.queue_df.head(20), use_container_width=True)
else:
    st.info("Click **Sample Queue** first.")
    st.stop()

queue_df = st.session_state.queue_df

st.markdown("---")
st.header("Results")

# --- Run FCFS ---
if run_fcfs:
    ordered, metrics = simulate_queue(
        queue_df,
        order_fn=fcfs_order,
        separation_minutes=float(separation),
        risk_threshold=float(risk_threshold),
        risk_penalty_minutes=float(risk_penalty),
    )

    st.subheader("FCFS Metrics")
    st.json(metrics_to_dict(metrics))

    show_cols = [
        "MONTH", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
        "DEP_MINUTES", "DISTANCE", "DEPARTURE_DELAY", "DELAYED_15",
        "pred_delay_risk", "READY_TIME",
        "assigned_time", "idle_time", "waiting_time", "total_delay_proxy"
    ]
    st.dataframe(ordered[show_cols].head(200), use_container_width=True)

# --- Run Intelligent ---
if run_intel:
    intel_df = queue_df.copy()
    intel_df["READY_TIME"] = intel_df["DEP_MINUTES"] + intel_df["DEPARTURE_DELAY"].clip(lower=0)

    ordered, metrics = simulate_queue(
        intel_df,
        order_fn=lambda d: intelligent_order(d, separation_minutes=float(separation)),
        separation_minutes=float(separation),
        risk_threshold=float(risk_threshold),
        risk_penalty_minutes=float(risk_penalty),
    )

    st.subheader("Intelligent Metrics")
    st.json(metrics_to_dict(metrics))

    show_cols = [
        "MONTH", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
        "DEP_MINUTES", "DISTANCE", "DEPARTURE_DELAY", "DELAYED_15",
        "pred_delay_risk", "READY_TIME",
        "assigned_time", "idle_time", "waiting_time", "total_delay_proxy"
    ]
    st.dataframe(ordered[show_cols].head(200), use_container_width=True)
