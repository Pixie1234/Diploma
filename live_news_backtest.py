#!/usr/bin/env python3
"""Live news sentiment backtest (best-effort real news)

Backtests Open/Close log-return prediction, with optional sentiment fusion using:
  - Alpha Vantage NEWS_SENTIMENT for a historical time window (uses time_from/time_to)
  - Finlight live articles (optional; only those returned will be used)

Runs sentiment inference once for all fetched articles, then aggregates per target day.
"""

import os
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES, OPEN_IDX, CLOSE_IDX
from lstm_model import load_or_train
from evaluation import evaluate_predictions
from sentiment2 import load_nlp, finbert_sentiment, roberta_sentiment, fuse_sentiment, apply_sentiment_fusion, compute_total_bias
from sp500 import load_sp500


@dataclass
class Article:
    bias: float
    publish_time: datetime | None


def _parse_alpha_time(time_str: str) -> datetime | None:
    time_str = (time_str or "").strip()
    if not time_str:
        return None
    try:
        # Alpha Vantage time formats (sometimes includes 'T'):
        # - YYYYMMDDHHMMSS
        # - YYYYMMDDTHHMMSS
        for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d%H%M%S"):
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                pass
        return None
    except Exception:
        return None


def _clean_text(s: str, max_len: int) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s[:max_len]


def _title_tokens(company: str) -> list[str]:
    parts = [t.strip().upper() for t in str(company).replace("-", " ").split() if len(t.strip()) >= 2]
    generic = {
        "COMPANY",
        "CO",
        "CORP",
        "CORPORATION",
        "INC",
        "LIMITED",
        "LTD",
        "PLC",
        "HOLDINGS",
        "GROUP",
        "TECHNOLOGIES",
        "TECHNOLOGY",
        "SERVICES",
        "SYSTEMS",
        "SYSTEM",
        "TRUST",
        "CAPITAL",
    }
    return [p for p in parts if p not in generic]


def _is_finlight_relevant(title: str, symbol_upper: str, company_tokens: list[str]) -> bool:
    t = (title or "").upper()
    if symbol_upper in t:
        return True
    return any(tok in t for tok in company_tokens)


def _fetch_alpha_articles(symbol: str, api_key: str, start_dt_utc: datetime, end_dt_utc: datetime, limit: int) -> list[dict]:
    import requests

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": api_key,
        "limit": limit,
        "sort": "EARLIEST",
        "time_from": start_dt_utc.strftime("%Y%m%dT%H%M"),
        "time_to": end_dt_utc.strftime("%Y%m%dT%H%M"),
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json() if resp is not None else {}
    return (data.get("feed", []) or [])


def _fetch_finlight_articles(symbol: str, finlight_key: str, company: str | None, limit: int) -> list[object]:
    # Best-effort: Finlight doesn't provide a reliable historical window here.
    from finlight_client import FinlightApi, ApiConfig
    from finlight_client.models import GetArticlesParams

    client = FinlightApi(ApiConfig(api_key=finlight_key))
    params = GetArticlesParams(query=symbol, language="en", limit=limit)
    articles = client.articles.fetch_articles(params)
    return list(articles.articles or [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--alpha-key", default=os.environ.get("ALPHA_VANTAGE_KEY", ""))
    parser.add_argument("--finlight-key", default=os.environ.get("FINLIGHT_API_KEY", ""))
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    days_to_backtest = args.days
    alpha_key = args.alpha_key
    finlight_key = args.finlight_key

    # If not provided via env/args, load from Streamlit secrets (best-effort).
    if not alpha_key or not finlight_key:
        try:
            import tomllib

            secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
                alpha_key = alpha_key or secrets.get("ALPHA_VANTAGE_KEY", "")
                finlight_key = finlight_key or secrets.get("FINLIGHT_API_KEY", "")
        except Exception:
            pass

    model_path = args.model_path or f"models/{symbol}_lstm_ohlcv_indicators_v9.h5"

    ctx = prepare_data(symbol)
    X_test = ctx["X_test"]
    y_test = ctx["y_test"]

    if len(X_test) < days_to_backtest:
        raise SystemExit(f"Not enough test samples: have {len(X_test)}, need {days_to_backtest}")

    # Backtest last N samples from the test split.
    bt_start = len(X_test) - days_to_backtest
    X_bt = X_test[bt_start:]
    y_bt = y_test[bt_start:]

    # Map each sample index in X_test to its target date.
    # X is built such that sample j corresponds to features index i=SEQ_LEN+j.
    # y_test starts at val_end in X (see data_pipeline.split logic).
    val_end = int(len(ctx["X"]) * 0.85)
    j0 = val_end + bt_start

    # dates_features length == features length.
    target_dates = [ctx["dates_features"][SEQ_LEN + (j0 + k)] for k in range(days_to_backtest)]

    model, _ = load_or_train(
        ctx["X_train"], ctx["y_train"],
        ctx["X_val"], ctx["y_val"],
        model_path,
    )

    # LSTM predictions in scaled space.
    y_pred_both = model.predict(X_bt, verbose=0)

    # y has shape (n, 2) with order: [Open_return, Close_return].
    # CLOSE_IDX=3 is a feature/scaler column index for inverse-transform.
    y_bt_close = y_bt[:, 1]
    y_pred_close_scaled = y_pred_both[:, 1]

    m_close_lstm, _, yt_close, yp_close = evaluate_predictions(
        y_bt_close, y_pred_close_scaled,
        ctx["scaler"], CLOSE_IDX, "Close"
    )

    # Load NLP models once.
    fin_tok, fin_mod, roberta = load_nlp()

    company = symbol
    try:
        df_sp500 = load_sp500()
        row = df_sp500[df_sp500["Symbol"] == symbol]
        if len(row) > 0:
            company = str(row.iloc[0]["Security"])
    except Exception:
        company = symbol

    company_tokens = _title_tokens(company)
    symbol_upper = symbol

    # Determine news fetch window in UTC.
    start_date = min(target_dates).date()
    end_date = max(target_dates).date()
    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    all_articles: list[Article] = []

    # Alpha Vantage historical window.
    if alpha_key:
        feed = _fetch_alpha_articles(symbol_upper, alpha_key, start_dt, end_dt, limit=1000)
        for art in feed:
            title = _clean_text(art.get("title", "No title") or "No title", 140)
            summary = _clean_text(art.get("summary", "") or "", 600)
            publish_time = _parse_alpha_time(art.get("time_published", ""))

            # Relevance filter (title-only).
            if not _is_finlight_relevant(title, symbol_upper, company_tokens):
                continue

            text = f"{title} {summary}".strip()
            fin_l, fin_s = finbert_sentiment(text, fin_tok, fin_mod)
            rob_l, rob_s = roberta_sentiment(text, roberta)
            bias = fuse_sentiment(fin_l, fin_s, rob_l, rob_s)
            all_articles.append(Article(bias=bias, publish_time=publish_time))

    # Finlight best-effort supplement (no guaranteed historical window).
    if finlight_key:
        try:
            fin_articles = _fetch_finlight_articles(symbol_upper, finlight_key, company, limit=20)
            for art in fin_articles:
                title = getattr(art, "title", "") or ""
                if not _is_finlight_relevant(title, symbol_upper, company_tokens):
                    continue

                summary_raw = getattr(art, "summary", None) or getattr(art, "content", None) or ""
                title_clean = " ".join(str(title).replace("\n", " ").replace("\r", " ").split())[:140]
                summary_clean = " ".join(str(summary_raw).replace("\n", " ").replace("\r", " ").split())[:200]
                text = f"{title_clean} {summary_clean}".strip()

                pub_date = getattr(art, "publishDate", None) or getattr(art, "publishedAt", None) or getattr(art, "published_at", None) or getattr(art, "publishDate", None)
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    except Exception:
                        pub_date = None
                if pub_date is None:
                    publish_time = None
                else:
                    publish_time = pub_date

                fin_l, fin_s = finbert_sentiment(text, fin_tok, fin_mod)
                rob_l, rob_s = roberta_sentiment(text, roberta)
                bias = fuse_sentiment(fin_l, fin_s, rob_l, rob_s)
                all_articles.append(Article(bias=bias, publish_time=publish_time))
        except Exception:
            # Ignore Finlight failures during backtest.
            pass

    # Precompute base close series for recent_returns.
    close_vals = ctx["df"]["Close"].values

    y_pred_close_fused = []
    total_biases = []
    impacts = []
    articles_per_day = []
    for k in range(days_to_backtest):
        target_dt = target_dates[k].to_pydatetime() if hasattr(target_dates[k], "to_pydatetime") else target_dates[k]
        # reference_time at end-of-day UTC (best-effort)
        ref_time = datetime(target_dt.year, target_dt.month, target_dt.day, 23, 59, tzinfo=timezone.utc)

        # Recency window: recent returns up to the base day (best-effort mapping).
        # base day in df is features index i=SEQ_LEN+j
        base_day_idx = (SEQ_LEN + (j0 + k))
        base_day_idx = int(max(1, min(base_day_idx, len(close_vals) - 1)))
        start_idx = max(0, base_day_idx - 25)
        recent_log_returns = np.diff(np.log(close_vals[start_idx:base_day_idx + 1]))

        # Articles known up to this day.
        known = [a for a in all_articles if (a.publish_time is not None and a.publish_time <= ref_time)]
        if known:
            total_bias = compute_total_bias(
                [a.bias for a in known],
                publish_times=[a.publish_time for a in known],
                reference_time=ref_time,
            )
        else:
            total_bias = 0.0

        total_biases.append(total_bias)
        articles_per_day.append(len(known))

        # LSTM predicted close return (log return).
        o_ret = 0.0  # open fusion not evaluated here
        c_ret = float(yp_close[k])

        fused = apply_sentiment_fusion(
            total_bias=total_bias,
            open_returns=[o_ret],
            close_returns=[c_ret],
            last_open=1.0,
            last_close=1.0,
            days_to_predict=1,
            recent_returns=recent_log_returns,
            forecast_decay=10.0,
        )
        impact_close = fused["impact_curve"][0]
        impacts.append(impact_close)
        fused_close_ret = c_ret + impact_close
        y_pred_close_fused.append(fused_close_ret)

    y_pred_close_fused = np.array(y_pred_close_fused, dtype=float)

    # Evaluate fused close returns.
    # Use evaluate_predictions by feeding scaled arrays back: easiest is to compute metrics ourselves.
    # For now, reuse sklearn-style metrics by calling evaluate_predictions with identity transform.
    # Since evaluate_predictions expects scaled inputs, we compute metrics directly.
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(yt_close, y_pred_close_fused)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(yt_close, y_pred_close_fused))
    ss_res = np.sum((yt_close - y_pred_close_fused) ** 2)
    ss_tot = np.sum((yt_close - np.mean(yt_close)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    y_true_dir = (yt_close > 0).astype(int)
    y_pred_dir = (y_pred_close_fused > 0).astype(int)
    from sklearn.metrics import accuracy_score, f1_score
    dir_acc = accuracy_score(y_true_dir, y_pred_dir)
    f1 = f1_score(y_true_dir, y_pred_dir, zero_division=0)

    m_fused = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Direction": dir_acc,
        "F1": f1,
    }

    print("\n=== BACKTEST CLOSE (last N test samples) ===")
    print("LSTM-only:", m_close_lstm)
    print("Fused (sentiment best-effort):", m_fused)
    print(f"Articles fetched: {len(all_articles)}")
    print(
        "Sentiment stats (per day): "
        f"avg_articles={np.mean(articles_per_day):.1f}, "
        f"total_bias_mean={np.mean(total_biases):+.4f}, "
        f"total_bias_std={np.std(total_biases):.4f}, "
        f"total_bias_min={np.min(total_biases):+.4f}, "
        f"total_bias_max={np.max(total_biases):+.4f}, "
        f"impact_mean={np.mean(impacts):+.5f}, "
        f"impact_std={np.std(impacts):.5f}, "
        f"impact_min={np.min(impacts):+.5f}, "
        f"impact_max={np.max(impacts):+.5f}"
    )


if __name__ == "__main__":
    # Allow running as: python live_news_backtest.py --symbol MMM --days 30
    main()
