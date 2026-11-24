"""
Replays held-out transactions one at a time, scores each with the trained
Isolation Forest, and appends the result to data/live_feed.csv.

The Streamlit dashboard just needs to poll/re-read data/live_feed.csv -
no server, no ports.

Usage:
    python stream_simulator.py [--interval 0.5] [--limit 2000]

Run train_model.py first - this script depends on the model, scaler,
threshold and held-out rows it produces in models/.
"""

import argparse
import csv
import pathlib
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LIVE_FEED_PATH = ROOT / "data" / "live_feed.csv"

FIELDNAMES = ["transaction_id", "timestamp", "amount", "risk_score", "is_flagged", "actual_class"]


def build_features(df: pd.DataFrame, scaler):
    v_cols = [c for c in df.columns if c.startswith("V")]
    amount_scaled = scaler.transform(df[["Amount"]])
    return np.hstack([df[v_cols].to_numpy(), amount_scaled])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between transactions")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N transactions")
    args = parser.parse_args()

    model = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    scaler = joblib.load(MODELS_DIR / "amount_scaler.pkl")
    config = joblib.load(MODELS_DIR / "config.pkl")
    threshold = config["threshold"]

    source_df = pd.read_csv(MODELS_DIR / "stream_source.csv")
    if args.limit:
        source_df = source_df.iloc[: args.limit]

    features = build_features(source_df, scaler)
    scores = -model.decision_function(features)

    LIVE_FEED_PATH.parent.mkdir(exist_ok=True)
    with open(LIVE_FEED_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    print(f"Streaming {len(source_df)} transactions to {LIVE_FEED_PATH}")
    print(f"Threshold: {threshold:.4f}  (interval={args.interval}s)")

    start_time = datetime.now()
    with open(LIVE_FEED_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for i, (row, score) in enumerate(zip(source_df.itertuples(), scores)):
            timestamp = start_time + timedelta(seconds=i * args.interval)
            writer.writerow(
                {
                    "transaction_id": i,
                    "timestamp": timestamp.isoformat(),
                    "amount": round(row.Amount, 2),
                    "risk_score": round(float(score), 6),
                    "is_flagged": bool(score >= threshold),
                    "actual_class": int(row.Class),
                }
            )
            f.flush()
            time.sleep(args.interval)

    print("Stream finished.")


if __name__ == "__main__":
    main()
