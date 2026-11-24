"""
Trains an Isolation Forest on the PaySim synthetic mobile-money dataset.

Usage:
    python train_model_paysim.py

Expects data/paysim.csv (columns: step, type, amount, nameOrig,
oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest,
isFraud, isFlaggedFraud).

Feature choice: PaySim's fraud generator always drains the sender's account
to exactly zero (oldbalanceOrg == amount and newbalanceOrig == 0 afterwards).
This occurs in 0 of the 6.35M legitimate rows, so it is an almost perfect
unsupervised signal - captured here as the single binary feature
`exact_drain`. Richer feature sets (adding amount, balances, destination
account velocity, hour-of-day) were tried and all performed *worse*:
Isolation Forest splits on random features, and once wide-range continuous
features are mixed in, splits get "spent" on noise instead of this feature,
diluting the score (PR-AUC dropped from ~0.98 to ~0.01-0.17 in testing).
So the fraud-scoring model intentionally uses just exact_drain.

nameOrig (sender) is almost always unique per row in PaySim (~6.35M unique
senders out of 6.36M rows), so sender-side velocity carries no signal.
nameDest (receiver) repeats heavily (some accounts receive 100+ transfers),
so velocity features are computed on the receiving account instead - these
are NOT fed into the fraud model (same dilution problem) but are saved for
the dashboard's velocity-tracking panel (Objective 3 / Data Analytics Core).

Splits chronologically by `step` into train/test so the replayed stream is
unseen data. isFraud is excluded from training features and used only to
evaluate the model afterwards.
"""

import pathlib
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "paysim.csv"
MODELS_DIR = ROOT / "models"
TRAIN_FRACTION = 0.7

MODEL_FEATURES = ["exact_drain"]

# Demo stream sizing: the full test split is ~1.9M rows at 0.24% fraud, which
# would take hours to replay and show few alerts early on. The curated demo
# stream instead targets a fixed row count with a visible fraud ratio, so a
# default `python stream_simulator_paysim.py` run finishes in a few minutes
# and actually shows flagged alerts.
DEMO_STREAM_ROWS = 600
DEMO_FRAUD_FRACTION = 0.08


def load_data(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            "Download the PaySim dataset CSV and place it there as paysim.csv."
        )
    df = pd.read_csv(path)
    return df.sort_values("step", kind="stable").reset_index(drop=True)


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["exact_drain"] = (
        (df["oldbalanceOrg"] == df["amount"]) & (df["newbalanceOrig"] == 0)
    ).astype(int)
    df["dest_untouched"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)
    df["is_transfer_or_cashout"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)

    # Velocity features - dashboard display only, not fed into the model (see module docstring).
    dest_group = df.groupby("nameDest")
    df["dest_prior_txn_count"] = dest_group.cumcount()
    df["dest_prior_amt_sum"] = dest_group["amount"].cumsum() - df["amount"]

    return df


def compute_threshold(model: IsolationForest) -> float:
    """Midpoint between the score IsolationForest assigns to exact_drain=0 vs =1.

    This is a single binary feature, so there are only two possible scores.
    Using their midpoint as the cut avoids the quantile-threshold instability
    that shows up once scores are no longer continuous (see train_model.py
    for the analogous continuous-feature version on the ULB dataset).
    """
    score_0, score_1 = -model.decision_function(np.array([[0], [1]]))
    return float((score_0 + score_1) / 2)


def benchmark_latency(model: IsolationForest, n_calls: int = 1000, n_warmup: int = 50) -> float:
    """Average wall-clock time to score a single transaction, in milliseconds.

    Scores one row at a time (not batched) since that's how a transaction
    "as it arrives" would actually be scored in a live stream - this is what
    "real-time" latency means here. A short warmup is discarded first since
    the first few calls pay a one-off cost (far slower than steady state)
    that wouldn't reflect a running service.
    """
    sample = np.array([[0]])
    for _ in range(n_warmup):
        model.decision_function(sample)
    start = time.perf_counter()
    for _ in range(n_calls):
        model.decision_function(sample)
    elapsed = time.perf_counter() - start
    return (elapsed / n_calls) * 1000


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    df = add_engineered_columns(df)

    split_idx = int(len(df) * TRAIN_FRACTION)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")
    print(f"Train fraud rate: {train_df['isFraud'].mean():.4%}")
    print(f"Test fraud rate:  {test_df['isFraud'].mean():.4%}")

    X_train = train_df[MODEL_FEATURES].to_numpy()
    X_test = test_df[MODEL_FEATURES].to_numpy()

    model = IsolationForest(n_estimators=200, contamination=0.0013, random_state=42, n_jobs=-1)
    model.fit(X_train)

    threshold = compute_threshold(model)
    test_scores = -model.decision_function(X_test)
    test_pred = (test_scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["isFraud"], test_pred, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(test_df["isFraud"], test_scores)
    latency_ms = benchmark_latency(model)

    print("\n--- Test set evaluation (isFraud used only for reporting) ---")
    print(f"Threshold:  {threshold:.6f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1:         {f1:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print(f"Flagged:    {test_pred.sum()}  (true fraud: {test_df['isFraud'].sum()})")
    print(f"Scoring latency: {latency_ms:.4f} ms/transaction (single-row decision_function call)")

    joblib.dump(model, MODELS_DIR / "isolation_forest_paysim.pkl")
    joblib.dump({"threshold": threshold}, MODELS_DIR / "paysim_config.pkl")

    # The full test split (~1.9M rows, 0.24% fraud) is what the metrics above
    # are computed on. For the live demo stream we curate a small, fixed-size
    # sample with a visible fraud ratio instead - the model/threshold are
    # unchanged, this just makes a default demo run finish in a few minutes
    # and actually show flagged alerts.
    fraud_pool = test_df[test_df["isFraud"] == 1]
    legit_pool = test_df[test_df["isFraud"] == 0]
    n_fraud = min(len(fraud_pool), int(DEMO_STREAM_ROWS * DEMO_FRAUD_FRACTION))
    n_legit = DEMO_STREAM_ROWS - n_fraud
    fraud_sample = fraud_pool.sample(n=n_fraud, random_state=42)
    legit_sample = legit_pool.sample(n=n_legit, random_state=42)
    stream_df = pd.concat([fraud_sample, legit_sample]).sort_values("step", kind="stable")

    keep_cols = [
        "step", "nameOrig", "nameDest", "amount", "type",
        "exact_drain", "dest_untouched", "is_transfer_or_cashout",
        "dest_prior_txn_count", "dest_prior_amt_sum", "isFraud",
    ]
    stream_df[keep_cols].to_csv(MODELS_DIR / "paysim_stream_source.csv", index=False)

    print(f"\nSaved model, threshold and stream source ({len(stream_df)} rows) to {MODELS_DIR}")


if __name__ == "__main__":
    main()
