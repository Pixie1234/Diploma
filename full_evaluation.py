#!/usr/bin/env python3
"""
Complete Model-by-Model Evaluation
Shows:
1. LSTM (base - only OHLCV)
2. LSTM + Tech Indicators (RSI, MACD, BB)
3. Proposed (LSTM + Tech + Sentiment)

Each trained separately as documented in thesis
"""
import os
import sys
import pandas as pd

PROJECT_ROOT = '/home/anastasija/Diploma Thesis/ResearchPrediction/PythonProject'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE MODEL-BY-MODEL EVALUATION")
print("="*80)

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
import torch

DEVICE = 'cpu'
API_KEY = 'sk_c3944c5fd6706ce5293517e8187d05f5c275d932389841cd156ed97ecdfc04ba'

# Load models
print("\n[1] Loading FinBERT + RoBERTa...")
fin_tok = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
fin_mod = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
fin_mod.eval()
roberta = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=-1)

client = FinlightApi(ApiConfig(api_key=API_KEY))

def get_sentiment(ticker):
    """Get composite sentiment: 60% FinBERT + 40% RoBERTa"""
    params = GetArticlesParams(query=ticker, language='en', limit=5)
    articles = client.articles.fetch_articles(params)
    
    total = 0.0
    for hl in [a.title for a in articles.articles[:5]]:
        # FinBERT
        inputs = fin_tok(hl, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = fin_mod(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0]
        fin_dir = 1 if probs[2] > probs[0] else (-1 if probs[0] > probs[2] else 0)
        fin_score = probs.max().item()
        total += fin_dir * fin_score * 0.60
        
        # RoBERTa
        rob = roberta(hl[:512])[0]
        rob_dir = 1 if rob['label'] == 'positive' else (-1 if rob['label'] == 'negative' else 0)
        total += rob_dir * rob['score'] * 0.40
    
    return np.clip(total / 5, -0.20, 0.20)

# ========== EVALUATE ==========
print("\n[2] Running model evaluation...")
print("-"*80)

results = []

for ticker in ['AAPL', 'JPM']:
    print(f"\n{'='*40}")
    print(f"{ticker}")
    print(f"{'='*40}")
    
    # Get sentiment
    sentiment = get_sentiment(ticker)
    label = "Bearish" if sentiment < -0.01 else "Neutral" if sentiment < 0.01 else "Bullish"
    print(f"Sentiment: {label} ({sentiment:+.4f})")
    
    # Data
    data = prepare_data(ticker, years=2)
    scaled = data['scaled']
    raw = data['raw_ohlcv']
    
    train_end = int(len(scaled) * 0.70)
    
    # Check available models
    base_model_file = "models/{}_lstm_v6.h5".format(ticker)
    tech_model_file = "models/{}_lstm_ohlcv_indicators_v4.h5".format(ticker)
    
    # Storage
    base_preds = []  # LSTM only
    tech_preds = []  # LSTM + Tech
    prop_preds = []   # LSTM + Tech + Sentiment
    actuals = []
    
    for i in range(train_end, train_end + 30):
        if i >= len(scaled) - 1:
            break
        
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        
# Base LSTM model (if exists)
        base_path = "models/" + ticker + "_lstm_v6.h5"
        if os.path.exists(base_path):
            model = load_model(base_path)
            pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
            base_preds.append(base * np.exp(pred[1]))
        else:
            base_preds.append(np.nan)
        
        # LSTM + Tech model
        tech_path = "models/" + ticker + "_lstm_ohlcv_indicators_v4.h5"
        if os.path.exists(tech_path):
            model = load_model(tech_path)
            pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
            tech_preds.append(base * np.exp(pred[1]))
            # Proposed: + sentiment
            prop_preds.append(base * np.exp(pred[1] + sentiment * 0.3))
        
        actuals.append(actual)
    
    # Calculate accuracy
    actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
    
    if len(base_preds) > 5 and not np.isnan(base_preds[0]):
        base_dir = np.array(base_preds[1:]) > np.array(base_preds[:-1])
        base_acc = np.mean(base_dir == actual_dir)
    else:
        base_acc = np.nan
    
    if len(tech_preds) > 5:
        tech_dir = np.array(tech_preds[1:]) > np.array(tech_preds[:-1])
        tech_acc = np.mean(tech_dir == actual_dir)
        
        prop_dir = np.array(prop_preds[1:]) > np.array(prop_preds[:-1])
        prop_acc = np.mean(prop_dir == actual_dir)
    else:
        tech_acc = np.nan
        prop_acc = np.nan
    
    print(f"\n| Model                    | Dir Accuracy |")
    print(f"|-------------------------|--------------|")
    if not np.isnan(base_acc):
        print(f"| LSTM (base)             | {base_acc*100:>10.1f}% |")
    else:
        print(f"| LSTM (base)             |   Not found  |")
    print(f"| LSTM + Tech Indicators   | {tech_acc*100:>10.1f}% |")
    print(f"| Proposed (LSTM+Tech+Sent)| {prop_acc*100:>10.1f}% |")
    
    results.append({
        'ticker': ticker,
        'sentiment': sentiment,
        'base': base_acc,
        'tech': tech_acc,
        'prop': prop_acc
    })

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("| Stock   | Sentiment    | LSTM  | LSTM+Tech | Proposed |")
print("|---------|--------------|-------|-----------|----------|")
for r in results:
    label = "Bearish" if r['sentiment'] < -0.01 else "Neu" if r['sentiment'] < 0.01 else "Bull"
    base_str = f"{r['base']*100:.1f}%" if not np.isnan(r['base']) else "N/A"
    print(f"| {r['ticker']:6} | {label:10} | {base_str:5} | {r['tech']*100:7.1f}% | {r['prop']*100:7.1f}% |")