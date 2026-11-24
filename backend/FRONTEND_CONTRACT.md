# Backend -> Frontend contract

## How to run the backend

```
C:\Users\aryan\IDA\venv\Scripts\python.exe C:\Users\aryan\IDA\backend\train_model_paysim.py
C:\Users\aryan\IDA\venv\Scripts\python.exe C:\Users\aryan\IDA\backend\stream_simulator_paysim.py
```

`train_model_paysim.py` only needs to be run once (or whenever the model
should be retrained). It prints evaluation metrics and writes the model,
threshold, and a curated 600-row demo stream to `models/`.

`stream_simulator_paysim.py` is what the dashboard should run against. It
appends one scored transaction every 0.5s (configurable with `--interval`)
to `data\live_feed.csv`, for 600 rows (~5 minutes) by default. Flags
(`--interval 0.1`, `--limit 100`) exist for a faster/shorter run during
development.

## data\live_feed.csv schema

The dashboard should poll/re-read this file (it grows one row at a time).
Columns, in order:

| Column | Type | Meaning |
|---|---|---|
| `transaction_id` | int | Row index in the stream, starts at 0 |
| `timestamp` | ISO datetime string | Simulated arrival time |
| `name_orig` | string | Sending account ID (e.g. `C1833399833`) |
| `name_dest` | string | Receiving account ID (e.g. `C1855260133` or `M...` for merchants) |
| `amount` | float | Transaction amount |
| `type` | string | One of `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `dest_prior_txn_count` | int | How many prior transactions this destination account has received (velocity signal - use for the velocity panel) |
| `dest_prior_amt_sum` | float | Total amount this destination account has received before this transaction (velocity signal) |
| `risk_score` | float | Isolation Forest anomaly score, higher = more anomalous. Not bounded to [0,1] - typically ranges roughly -0.1 to +0.1 here |
| `is_flagged` | bool (`True`/`False` string) | Whether `risk_score` crossed the alert threshold |
| `actual_class` | int (0/1) | Ground truth fraud label - only for the evaluation panel, never display this as if the system "knew" it live |

Example rows:
```
transaction_id,timestamp,name_orig,name_dest,amount,type,dest_prior_txn_count,dest_prior_amt_sum,risk_score,is_flagged,actual_class
0,2026-11-24T17:19:10.821114,C1833399833,C1855260133,149146.5,CASH_OUT,15,3308433.42,-0.0,False,0
1,2026-11-24T17:19:10.871114,C1057819107,C97284144,10211.81,TRANSFER,0,0.0,0.064829,True,1
```

## Notes for the dashboard build

- `is_flagged` is already computed server-side - the dashboard doesn't need
  to re-threshold `risk_score` itself, though the original frontend spec's
  "adjustable threshold slider" can still re-filter using `risk_score` if
  you want that interactivity (just compare against a slider value instead
  of trusting `is_flagged` directly).
- The file is rewritten (header + truncated) every time `stream_simulator_paysim.py`
  is restarted, so the dashboard should handle the file shrinking/resetting
  between runs, not just growing.
- `dest_prior_txn_count` / `dest_prior_amt_sum` are what feed the
  velocity-tracking panel from the original spec - they are NOT inputs to
  the fraud model itself, just descriptive context computed alongside it.
- Expect roughly 8% of rows to have `actual_class=1` in this demo stream
  (curated on purpose - real-world fraud rate is ~0.13%, see
  `train_model_paysim.py` docstring for why).
