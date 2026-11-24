"""
Replays the curated PaySim demo stream one transaction at a time, scores
each with the trained Isolation Forest, and appends the result to
data/live_feed.csv.

The Streamlit dashboard just needs to poll/re-read data/live_feed.csv -
no server, no ports.

Usage:
    python stream_simulator_paysim.py [--interval 0.5] [--limit 2000]

Run train_model_paysim.py first - this script depends on the model,
threshold and curated stream rows it produces in models/.
"""

import argparse
import csv
import pathlib
import time
from datetime import datetime, timedelta

import joblib
import pandas as pd

from train_model_paysim import MODEL_FEATURES

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LIVE_FEED_PATH = ROOT / "data" / "live_feed.csv"

FIELDNAMES = [
    "transaction_id", "timestamp", "name_orig", "name_dest", "amount", "type",
    "dest_prior_txn_count", "dest_prior_amt_sum", "risk_score", "is_flagged", "actual_class",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between transactions")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N transactions")
    args = parser.parse_args()

    model = joblib.load(MODELS_DIR / "isolation_forest_paysim.pkl")
    config = joblib.load(MODELS_DIR / "paysim_config.pkl")
    threshold = config["threshold"]

    source_df = pd.read_csv(MODELS_DIR / "paysim_stream_source.csv")
    if args.limit:
        source_df = source_df.iloc[: args.limit]

    scores = -model.decision_function(source_df[MODEL_FEATURES].to_numpy())

    LIVE_FEED_PATH.parent.mkdir(exist_ok=True)
    with open(LIVE_FEED_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    print(f"Streaming {len(source_df)} transactions to {LIVE_FEED_PATH}")
    print(f"Threshold: {threshold:.6f}  (interval={args.interval}s)")

    start_time = datetime.now()
    with open(LIVE_FEED_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for i, (row, score) in enumerate(zip(source_df.itertuples(), scores)):
            timestamp = start_time + timedelta(seconds=i * args.interval)
            writer.writerow(
                {
                    "transaction_id": i,
                    "timestamp": timestamp.isoformat(),
                    "name_orig": row.nameOrig,
                    "name_dest": row.nameDest,
                    "amount": round(row.amount, 2),
                    "type": row.type,
                    "dest_prior_txn_count": int(row.dest_prior_txn_count),
                    "dest_prior_amt_sum": round(float(row.dest_prior_amt_sum), 2),
                    "risk_score": round(float(score), 6),
                    "is_flagged": bool(score >= threshold),
                    "actual_class": int(row.isFraud),
                }
            )
            f.flush()
            time.sleep(args.interval)

    print("Stream finished.")


if __name__ == "__main__":
    main()
