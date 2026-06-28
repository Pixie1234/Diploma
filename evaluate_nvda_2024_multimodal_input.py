#!/usr/bin/env python3
"""NVDA 2024 multimodal evaluation with sentiment as an input feature.

This version appends the daily real-news sentiment series to the model inputs,
then retrains LSTM and Informer on the augmented sequences. It avoids the
post-hoc sentiment cap used by the earlier fusion scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_pipeline import SEQ_LEN, OPEN_IDX, CLOSE_IDX, add_indicators, build_feature_matrix, load_price
from evaluation import evaluate_predictions
from evaluate_nvda_2024_same_setup import get_full_predictions, set_seed, train_base_informer
from lstm_model import train_lstm


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing split file: {path}")
    df = pd.read_csv(path)
    if "day" not in df.columns or "weighted_bias" not in df.columns:
        raise SystemExit(f"Split file must contain day and weighted_bias: {path}")
    df["day"] = pd.to_datetime(df["day"], errors="coerce", utc=True)
    return df.dropna(subset=["day"]).sort_values("day").reset_index(drop=True)


def _sentiment_lookup(df: pd.DataFrame) -> dict[pd.Timestamp, float]:
    lookup: dict[pd.Timestamp, float] = {}
    for _, row in df.iterrows():
        lookup[pd.Timestamp(row["day"]).normalize()] = float(row["weighted_bias"])
    return lookup


def _sentiment_windows(feature_dates: pd.DatetimeIndex, lookup: dict[pd.Timestamp, float]) -> np.ndarray:
    sentiment = np.array([float(lookup.get(pd.Timestamp(d).normalize(), 0.0)) for d in feature_dates], dtype=float)
    sentiment = np.clip(sentiment, -0.20, 0.20)
    windows = []
    for i in range(SEQ_LEN, len(sentiment)):
        windows.append(sentiment[i - SEQ_LEN : i])
    return np.asarray(windows, dtype=float)[..., None]


def _split_indices(all_dates: pd.DatetimeIndex, df: pd.DataFrame) -> np.ndarray:
    date_index = {pd.Timestamp(d).normalize(): i for i, d in enumerate(all_dates)}
    idx = [date_index.get(pd.Timestamp(d).normalize()) for d in df["day"]]
    return np.asarray([i for i in idx if i is not None], dtype=int)


def _build_ohlcv_dataset_train_scaled(raw_ohlcv: np.ndarray, train_idx: np.ndarray, seq_len: int = SEQ_LEN):
    price_rets = np.log(raw_ohlcv[1:] / (raw_ohlcv[:-1] + 1e-10))
    scaler = StandardScaler()

    if train_idx.size == 0:
        raise SystemExit("Empty training index set")

    train_fit_end = min(len(price_rets), seq_len + int(train_idx.max()) + 1)
    scaler.fit(price_rets[:train_fit_end, :5])
    scaled = scaler.transform(price_rets[:, :5])

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len : i, :])
        y.append(scaled[i, :2])

    return np.asarray(X), np.asarray(y), scaler


def main() -> int:
    parser = argparse.ArgumentParser(description="NVDA 2024 multimodal input-feature evaluation.")
    parser.add_argument("--years", type=int, default=10, help="Historical price window for context")
    parser.add_argument("--symbol", default="NVDA", help="Ticker symbol")
    parser.add_argument("--train", default="data/news_price_merged/full_deduped_scored_daily_train.csv")
    parser.add_argument("--eval", default="data/news_price_merged/full_deduped_scored_daily_eval.csv")
    parser.add_argument("--test", default="data/news_price_merged/full_deduped_scored_daily_test.csv")
    args = parser.parse_args()

    set_seed(42)
    symbol = args.symbol.upper()

    train_df = _load_split(Path(args.train))
    eval_df = _load_split(Path(args.eval))
    test_df = _load_split(Path(args.test))
    full_news = pd.concat([train_df, eval_df, test_df], ignore_index=True).sort_values("day").reset_index(drop=True)
    sentiment_lookup = _sentiment_lookup(full_news)

    price_df = load_price(symbol, years=args.years)
    price_df = add_indicators(price_df)
    _, raw_ohlcv = build_feature_matrix(price_df)
    feature_dates = pd.to_datetime(price_df.index[1:], utc=True)
    all_dates = feature_dates[SEQ_LEN:]

    train_idx = _split_indices(all_dates, train_df)
    eval_idx = _split_indices(all_dates, eval_df)
    test_idx = _split_indices(all_dates, test_df)

    base_X, base_y, base_scaler = _build_ohlcv_dataset_train_scaled(raw_ohlcv, train_idx, seq_len=SEQ_LEN)
    sentiment_X = _sentiment_windows(feature_dates, sentiment_lookup)
    multimodal_X = np.concatenate([base_X, sentiment_X], axis=-1)

    base_lstm_model = train_lstm(
        base_X[train_idx], base_y[train_idx], base_X[eval_idx], base_y[eval_idx], model_path=None, n_features=base_X.shape[-1]
    )
    multimodal_lstm_model = train_lstm(
        multimodal_X[train_idx],
        base_y[train_idx],
        multimodal_X[eval_idx],
        base_y[eval_idx],
        model_path=None,
        n_features=multimodal_X.shape[-1],
    )

    base_inf_model = train_base_informer(base_X[train_idx], base_y[train_idx], base_X[eval_idx], base_y[eval_idx])
    multimodal_inf_model = train_base_informer(
        multimodal_X[train_idx],
        base_y[train_idx],
        multimodal_X[eval_idx],
        base_y[eval_idx],
    )

    base_lstm_full = get_full_predictions(base_lstm_model, base_X)
    multimodal_lstm_full = get_full_predictions(multimodal_lstm_model, multimodal_X)
    base_inf_full = get_full_predictions(base_inf_model, base_X)
    multimodal_inf_full = get_full_predictions(multimodal_inf_model, multimodal_X)

    proposed_full = 0.5 * (multimodal_lstm_full + multimodal_inf_full)

    rows_open = []
    for name, y_true, y_pred, scaler in [
        ("LSTM (OHLCV)", base_y[test_idx][:, 0], base_lstm_full[test_idx, 0], base_scaler),
        ("LSTM + Sentiment Input", base_y[test_idx][:, 0], multimodal_lstm_full[test_idx, 0], base_scaler),
        ("Informer (OHLCV)", base_y[test_idx][:, 0], base_inf_full[test_idx, 0], base_scaler),
        ("Informer + Sentiment Input", base_y[test_idx][:, 0], multimodal_inf_full[test_idx, 0], base_scaler),
        ("Proposed (LSTM + Informer + Sentiment Input)", base_y[test_idx][:, 0], proposed_full[test_idx, 0], base_scaler),
    ]:
        metrics, _, _, _ = evaluate_predictions(y_true, y_pred, scaler, OPEN_IDX, "Open")
        rows_open.append((name, metrics["Directional Accuracy"], metrics["MAE"]))

    rows_close = []
    for name, y_true, y_pred, scaler in [
        ("LSTM (OHLCV)", base_y[test_idx][:, 1], base_lstm_full[test_idx, 1], base_scaler),
        ("LSTM + Sentiment Input", base_y[test_idx][:, 1], multimodal_lstm_full[test_idx, 1], base_scaler),
        ("Informer (OHLCV)", base_y[test_idx][:, 1], base_inf_full[test_idx, 1], base_scaler),
        ("Informer + Sentiment Input", base_y[test_idx][:, 1], multimodal_inf_full[test_idx, 1], base_scaler),
        ("Proposed (LSTM + Informer + Sentiment Input)", base_y[test_idx][:, 1], proposed_full[test_idx, 1], base_scaler),
    ]:
        metrics, _, _, _ = evaluate_predictions(y_true, y_pred, scaler, CLOSE_IDX, "Close")
        rows_close.append((name, metrics["Directional Accuracy"], metrics["MAE"]))

    print("\nNVDA 2024 multimodal input-feature comparison")
    print(f"Test days: {len(test_idx)} | Eval days: {len(eval_idx)}")
    print("Sentiment appended as an input feature, not applied post-hoc")
    print("\n| Model | Open Directional Accuracy | Open MAE |")
    print("|---|---:|---:|")
    for name, da, mae in rows_open:
        print(f"| {name} | {da:.4f} | {mae:.8f} |")
    print("\n| Model | Close Directional Accuracy | Close MAE |")
    print("|---|---:|---:|")
    for name, da, mae in rows_close:
        print(f"| {name} | {da:.4f} | {mae:.8f} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
