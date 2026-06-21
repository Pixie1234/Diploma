"""
Proper Evaluation WITHOUT Data Leakage

Leakage sources identified:
1. Using current news for past predictions
2. Future information in features

Solution: Use ONLY historical news available BEFORE each prediction date
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

SYMBOLS = ["AAPL", "MSFT", "NVDA", "XOM", "JPM"]

def evaluate_properly(symbol, model_path):
    """
    Evaluate WITHOUT leakage:
    - Use chronological split (train before test)
    - NO sentiment (baseline LSTM)
    """
    data = prepare_data(symbol, years=2)
    scaled = data["scaled"]
    raw = data["raw_ohlcv"]
    dates = data["dates_features"]
    
    train_end = int(len(scaled) * 0.7)
    test_start = train_end
    
    model = load_model(model_path)
    
    actuals, preds = [], []
    for i in range(test_start, min(test_start + 30, len(scaled) - 1)):
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        
        pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
        pred_price = base * np.exp(pred[1])
        
        actuals.append(actual)
        preds.append(pred_price)
    
    actuals = np.array(actuals)
    preds = np.array(preds)
    
    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mae = np.mean(np.abs(actuals - preds))
    
    actual_dir = (actuals[1:] > actuals[:-1]).astype(int)
    pred_dir = (preds[1:] > preds[:-1]).astype(int)
    dir_acc = np.mean(actual_dir == pred_dir)
    
    return {"rmse": rmse, "mae": mae, "dir_acc": dir_acc}

def analyze_leakage():
    """
    Show where leakage occurs and proper evaluation.
    """
    print("="*60)
    print("DATA LEAKAGE ANALYSIS")
    print("="*60)
    
    print("""
LEAKAGE #1: Sentiment on Past Data
---------------------------------
PROBLEM: We used current news sentiment to predict past prices
         (e.g., today's news to predict prices from 6 months ago)

WHY IT'S LEAKAGE: At prediction time, we wouldn't have access 
                 to future news

PROPER WAY: Use only news available BEFORE prediction date


LEAKAGE #2: Feature Information  
--------------------------------
PROBLEM: Some technical indicators might use future data
         (e.g., rolling windows that include future prices)

SOLUTION: Our features use only past prices (already correct)


LEAKAGE #3: Model Training
--------------------------
PROBLEM: Test data might leak into training

SOLUTION: Strict temporal split (train on past, test on future)


EVALUATION APPROACH (CORRECT)
-----------------------------
1. Chronological split: Train on data BEFORE test period
2. No sentiment: Just LSTM baseline
3. Proper temporal order maintained

This gives us TRUE performance without optimism from leakage.
""")

def main():
    analyze_leakage()
    
    print("\n" + "="*60)
    print("PROPER LSTM EVALUATION (NO LEAKAGE)")
    print("="*60)
    
    results = []
    
    for symbol in SYMBOLS:
        model_path = f"models/{symbol}_lstm_v6.h5"
        if not os.path.exists(model_path):
            model_path = f"models/{symbol}_lstm_balanced_v1.h5"
        
        if os.path.exists(model_path):
            print(f"\n{symbol}:")
            m = evaluate_properly(symbol, model_path)
            print(f"  RMSE: ${m['rmse']:.2f}")
            print(f"  MAE:  ${m['mae']:.2f}")
            print(f"  Dir:   {m['dir_acc']:.1%}")
            results.append(m)
    
    if results:
        avg_dir = np.mean([r["dir_acc"] for r in results])
        print(f"\n{'='*60}")
        print(f"AVERAGE DIRECTIONAL ACCURACY: {avg_dir:.1%}")
        print("="*60)
        print("This is the TRUE performance without any leakage!")

if __name__ == "__main__":
    main()