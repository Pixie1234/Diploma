#!/usr/bin/env python3
"""
Model Comparison: Baseline vs LSTM+Tech vs Proposed
Using available models
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.insert(0, '.')
import numpy as np
from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE MODEL COMPARISON")
print("="*80)

# Sentiment setup
from finlight_client import FinlightApi, ApiConfig
API_KEY = 'sk_c3944c5fd6706ce5293517e8187d05f5c275d932389841cd156ed97ecdfc04ba'

DEVICE = 'cpu'
fin_tok = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
fin_mod = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
fin_mod.eval()
roberta = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=-1)
client = FinlightApi(ApiConfig(api_key=API_KEY))

# Get sentiment
def get_composite_sentiment(ticker):
    from finlight_client.models import GetArticlesParams
    params = GetArticlesParams(query=ticker, language='en', limit=3)
    articles = client.articles.fetch_articles(params)
    
    total = 0.0
    for hl in [a.title for a in articles.articles]:
        inputs = fin_tok(hl, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = fin_mod(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0]
        total += (1 if probs[2] > probs[0] else (-1 if probs[0] > probs[2] else 0)) * probs.max().item() * 0.60
        
        rob = roberta(hl[:512])[0]
        total += (1 if rob['label'] == 'positive' else (-1 if rob['label'] == 'negative' else 0)) * rob['score'] * 0.40
    
    return np.clip(total / 3, -0.20, 0.20)

# Evaluate
from finlight_client.models import GetArticlesParams
results = []

for ticker in ['AAPL', 'JPM']:
    print(f"\n{'='*40}")
    print(f"{ticker}")
    print(f"{'='*40}")
    
    # 1. Get sentiment
    sentiment = get_composite_sentiment(ticker)
    label = "Bearish" if sentiment < -0.01 else "Neutral" if sentiment < 0.01 else "Bullish"
    print(f"Sentiment: {label} ({sentiment:+.4f})")
    
    # 2. Load data & model
    data = prepare_data(ticker, years=2)
    scaled = data['scaled']
    raw = data['raw_ohlcv']
    train_end = int(len(scaled) * 0.65)  # Use 65% as thesis
    
    # LSTM+Tech model (the only one we have)
    model = load_model('models/{}_lstm_ohlcv_indicators_v4.h5'.format(ticker))
    
    # Test on 30 days
    base_preds = []
    sent_preds = []
    actuals = []
    
    for i in range(train_end, train_end + 30):
        if i >= len(scaled) - 1:
            break
        
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        pred = model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
        
        # Base prediction (LSTM+Tech baseline)
        base_preds.append(base * np.exp(pred[1]))
        # With sentiment (Proposed)
        sent_preds.append(base * np.exp(pred[1] + sentiment * 0.3))
        actuals.append(actual)
    
    # Directional accuracy
    actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
    
    base_dir = np.array(base_preds[1:]) > np.array(base_preds[:-1])
    base_acc = np.mean(base_dir == actual_dir)
    
    sent_dir = np.array(sent_preds[1:]) > np.array(sent_preds[:-1])
    sent_acc = np.mean(sent_dir == actual_dir)
    
    print(f"\n{'Model':<30} {'Dir Acc':>10}")
    print(f"{'-'*40}")
    print(f"{'LSTM (baseline)':<30} {'N/A':>10}")  # Not available - same model
    print(f"{'LSTM + Tech Indicators':<30} {base_acc*100:>10.1f}%")
    print(f"{'Proposed (LSTM+Tech+Sentiment)':<30} {sent_acc*100:>10.1f}%")
    
    results.append({'t': ticker, 'base': base_acc, 'sent': sent_acc, 'sent_val': sentiment})

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Stock':<10} {'Sentiment':<12} {'LSTM+Tech':>10} {'Proposed':>10}")
print(f"{'-'*50}")
for r in results:
    label = "Bearish" if r['sent_val'] < -0.01 else "Neu" if r['sent_val'] < 0.01 else "Bull"
    print(f"{r['t']:<10} {label:12} {r['base']*100:10.1f}% {r['sent']*100:10.1f}%")

print("\nNote: LSTM (baseline) not available - same model used as LSTM+Tech")
print("The thesis results documented in RESULTS.md are separate evaluations")