"""
Trains an Isolation Forest on the ULB creditcard.csv dataset.

Usage:
    python train_model.py

Expects data/creditcard.csv (download from Kaggle: "Credit Card Fraud
Detection" by the ULB Machine Learning Group) relative to the project root.

Splits chronologically (by the Time column) into train/test so the
"live stream" replayed later (stream_simulator.py) is data the model
never saw during training. Class is excluded from training features
and used only to evaluate the model afterwards.
"""

import pathlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
TRAIN_FRACTION = 0.7

# Demo stream sizing: the full test split is ~85k rows at ~0.17% fraud, which
# would take hours to replay by default. The curated demo stream instead
# targets a fixed row count with a visible fraud ratio, so a default
# `python stream_simulator.py` run finishes in a few minutes.
DEMO_STREAM_ROWS = 600
DEMO_FRAUD_FRACTION = 0.08


def load_and_split(path: pathlib.Path, train_fraction: float):
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            "Download 'creditcard.csv' from the Kaggle 'Credit Card Fraud "
            "Detection' dataset (ULB Machine Learning Group) and place it there."
        )
    df = pd.read_csv(path)
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_features(df: pd.DataFrame, scaler: RobustScaler, fit: bool):
    v_cols = [c for c in df.columns if c.startswith("V")]
    amount_scaled = (
        scaler.fit_transform(df[["Amount"]]) if fit else scaler.transform(df[["Amount"]])
    )
    features = df[v_cols].to_numpy()
    features = np.hstack([features, amount_scaled])
    return features


def benchmark_latency(
    model: IsolationForest, sample_row: np.ndarray, n_calls: int = 1000, n_warmup: int = 50
) -> float:
    """Average wall-clock time to score a single transaction, in milliseconds.

    A short warmup is discarded first since the first few calls pay a
    one-off cost that wouldn't reflect a running, already-loaded service.
    """
    sample = sample_row.reshape(1, -1)
    for _ in range(n_warmup):
        model.decision_function(sample)
    start = time.perf_counter()
    for _ in range(n_calls):
        model.decision_function(sample)
    elapsed = time.perf_counter() - start
    return (elapsed / n_calls) * 1000


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    train_df, test_df = load_and_split(DATA_PATH, TRAIN_FRACTION)
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")
    print(f"Train fraud rate: {train_df['Class'].mean():.4%}")
    print(f"Test fraud rate:  {test_df['Class'].mean():.4%}")

    scaler = RobustScaler()
    X_train = build_features(train_df, scaler, fit=True)
    X_test = build_features(test_df, scaler, fit=False)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.002,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # decision_function: lower = more abnormal. Flip sign so higher = more anomalous.
    train_scores = -model.decision_function(X_train)
    test_scores = -model.decision_function(X_test)

    threshold = np.quantile(train_scores, 1 - 0.002)
    test_pred = (test_scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["Class"], test_pred, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(test_df["Class"], test_scores)
    latency_ms = benchmark_latency(model, X_test[0])

    print("\n--- Test set evaluation (Class used only for reporting) ---")
    print(f"Threshold:  {threshold:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print(f"Scoring latency: {latency_ms:.4f} ms/transaction (single-row decision_function call)")

    joblib.dump(model, MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, MODELS_DIR / "amount_scaler.pkl")
    joblib.dump({"threshold": threshold}, MODELS_DIR / "config.pkl")

    # Curate a small, fixed-size demo stream with a visible fraud ratio instead
    # of dumping the full ~85k-row test split - keeps a default stream_simulator.py
    # run to a few minutes and guarantees it actually shows flagged alerts.
    fraud_pool = test_df[test_df["Class"] == 1]
    legit_pool = test_df[test_df["Class"] == 0]
    n_fraud = min(len(fraud_pool), int(DEMO_STREAM_ROWS * DEMO_FRAUD_FRACTION))
    n_legit = DEMO_STREAM_ROWS - n_fraud
    fraud_sample = fraud_pool.sample(n=n_fraud, random_state=42)
    legit_sample = legit_pool.sample(n=n_legit, random_state=42)
    stream_df = pd.concat([fraud_sample, legit_sample]).sort_values("Time", kind="stable")
    stream_df.to_csv(MODELS_DIR / "stream_source.csv", index=False)

    print(f"\nSaved model, scaler, threshold and stream source ({len(stream_df)} rows) to {MODELS_DIR}")


if __name__ == "__main__":
    main()
