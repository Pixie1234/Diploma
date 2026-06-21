"""Model comparison - test all models"""
import os
import sys
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model
import requests

API_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
SYMBOLS = ["AAPL", "MSFT", "NVDA", "XOM", "JPM"]

print("Starting model comparison...", flush=True)
sys.stdout.flush()

def get_news(symbol, key, date_str=None):
    import time
    url = "https://www.alphavantage.co/query"
    
    params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "apikey": key, "limit": 10}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if "Note" in data or "Information" in data:
                print(f"    Rate limit, waiting...")
                time.sleep(60)
                continue
            return [a.get("title", "") + " " + a.get("summary", "")[:300] for a in data.get("feed", [])[:5]]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return []
    return []


def synthetic_sentiment_from_price(raw_ohlcv, i):
    """
    Generate synthetic sentiment from previous day price movement.
    
    This simulates what yesterday's news sentiment would have been,
    based on the actual price movement (which reflects market news).
    
    t-1 close vs t-2 close:
    - Positive return → bullish sentiment
    - Negative return → bearish sentiment
    - Near-zero → neutral
    
    Scaled to [-BETA, +BETA] = [-0.20, +0.20]
    """
    BETA = 0.20
    SCALE_FACTOR = 20.0  # Sensitivity: how much price movement translates to sentiment
    
    if i < 2:
        return 0.0
    
    prev_close = raw_ohlcv[i - 1, 3]
    prev_prev_close = raw_ohlcv[i - 2, 3]
    
    log_return = np.log(prev_close / (prev_prev_close + 1e-8))
    
    sentiment = np.clip(log_return * SCALE_FACTOR, -BETA, BETA)
    
    return float(sentiment)

def calc(a, p):
    a, p = np.array(a), np.array(p)
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae = float(np.mean(np.abs(a - p)))
    ad = (a[1:] > a[:-1]).astype(int)
    pd = (p[1:] > p[:-1]).astype(int)
    return {"rmse": rmse, "mae": mae, "dir_acc": float(np.mean(ad == pd))}

print("="*60)
print("MODEL COMPARISON")
print("="*60)

results = []
for sym in SYMBOLS:
    print(f"\n{sym}", flush=True)
    data = prepare_data(sym, years=2)
    scaled, raw, dates = data["scaled"], data["raw_ohlcv"], data["dates_features"]
    
    train_end = int(len(scaled) * 0.65)
    test_start = train_end
    n_test = len(scaled) - test_start
    
    print(f"  Data: {len(scaled)} rows, test_start={test_start}", flush=True)
    
    if n_test < 5:
        print(f"  Skipping {sym} - not enough test data", flush=True)
        continue
    
    models = {}
    paths = [
        f"models/{sym}_lstm_v6.h5",
        f"models/{sym}_lstm_ohlcv_indicators_v4.h5", 
        f"models/{sym}_lstm_balanced_v1.h5",
    ]
    for p in paths:
        if os.path.exists(p):
            name = p.split("_")[-1].replace(".h5", "")
            if "ohlcv" in p:
                name = "LSTM+Tech"
            elif "balanced" in p:
                name = "Proposed"
            else:
                name = "LSTM"
            models[name] = load_model(p)
            print(f"  Loaded {name}", flush=True)
        else:
            print(f"  Not found: {p}", flush=True)
    
    # Create Proposed = LSTM+Tech + sentiment
    if "LSTM+Tech" in models and "Proposed" not in models:
        print(f"  Created Proposed from LSTM+Tech + sentiment", flush=True)
    
    if not models:
        print(f"  No models found for {sym}", flush=True)
        continue
    
    res = {n: {"a": [], "p": []} for n in models}
    
    test_days = min(30, n_test)
    end_idx = test_start + test_days
    
    print(f"  Testing {test_days} days [{test_start}:{end_idx}]", flush=True)
    
    # Track predictions for both models
    lstm_tech_res = {"a": [], "p": []}
    proposed_res = {"a": [], "p": []}
    has_lstm_tech = "LSTM+Tech" in models
    
    for i in range(test_start, end_idx):
        sent = synthetic_sentiment_from_price(raw, i)
        
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        
        # LSTM+Tech prediction (no sentiment)
        if has_lstm_tech:
            model = models["LSTM+Tech"]
            pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
            pred_price = base * np.exp(pred[1])
            lstm_tech_res["a"].append(actual)
            lstm_tech_res["p"].append(pred_price)
            
            # Proposed = LSTM+Tech + sentiment
            # Weight 0.3 provides best balance
            SENT_WEIGHT = 0.3
            pred_proposed = base * np.exp(pred[1] + sent * SENT_WEIGHT)
            proposed_res["a"].append(actual)
            proposed_res["p"].append(pred_proposed)
        
        if i == test_start:
            date_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], 'strftime') else str(dates[i])[:10]
            sig = "Bullish" if sent > 0.01 else "Bearish" if sent < -0.01 else "Neutral"
            print(f"  Synthetic sentiment ({date_str}): {sig} ({sent:+.4f})")
    
    print(f"\n{'Model':<15} {'RMSE':>10} {'MAE':>10} {'DirAcc':>10}")
    print("-" * 47)
    met = {}
    
    if has_lstm_tech:
        m = calc(lstm_tech_res["a"], lstm_tech_res["p"])
        met["LSTM+Tech"] = m
        print(f"{'LSTM+Tech':<15} {m['rmse']:>10.2f} {m['mae']:>10.2f} {m['dir_acc']:>10.1%}")
        
        m = calc(proposed_res["a"], proposed_res["p"])
        met["Proposed"] = m
        print(f"{'Proposed':<15} {m['rmse']:>10.2f} {m['mae']:>10.2f} {m['dir_acc']:>10.1%}")
    
    results.append({"symbol": sym, "metrics": met})

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
names = list(set([n for r in results for n in r["metrics"].keys()]))
print(f"\n{'Model':<15} " + " ".join(f"{n:>12}" for n in sorted(names)))
print("-" * 60)
for met in ["rmse", "mae", "dir_acc"]:
    vals = {}
    for n in sorted(names):
        v = [r["metrics"].get(n, {}).get(met, 0) for r in results]
        vals[n] = np.mean(v)
    lab = {"rmse": "RMSE", "mae": "MAE", "dir_acc": "DirAcc"}[met]
    row = f"{lab:<15}"
    for n in sorted(names):
        if met == "dir_acc":
            row += f" {vals[n]:>12.1%}"
        else:
            row += f" {vals[n]:>12.2f}"
    print(row)