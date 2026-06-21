#!/usr/bin/env python3
"""
Show exact predictions with dates and prices
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
print("EXACT PREDICTIONS WITH DATES AND PRICES")
print("="*80)

from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT + RoBERTa
DEVICE = 'cpu'
fin_tok = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
fin_mod = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
fin_mod.eval()
roberta = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=-1)

# Get current sentiment from Finlight
client = FinlightApi(ApiConfig(api_key='sk_c3944c5fd6706ce5293517e8187d05f5c275d932389841cd156ed97ecdfc04ba'))

def get_sentiment(text):
    inputs = fin_tok(text, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = fin_mod(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0]
    labels = ['Bearish', 'Neutral', 'Bullish']
    fin_l = labels[torch.argmax(probs).item()]
    fin_s = probs[torch.argmax(probs)].item()
    rob = roberta(text[:512])[0]
    rob_l, rob_s = rob['label'], rob['score']
    
    def dir(l):
        l = l.upper()
        if 'BULLISH' in l or 'POSITIVE' in l: return 1
        if 'BEARISH' in l or 'NEGATIVE' in l: return -1
        return 0
    
    fused = dir(fin_l) * fin_s * 0.6 + dir(rob_l) * rob_s * 0.4
    return np.clip(fused, -0.2, 0.2)

# Get news
for ticker in ['AAPL', 'JPM']:
    print(f"\n{'='*40}")
    print(f"{ticker}")
    print(f"{'='*40}")
    
    params = GetArticlesParams(query=ticker, language='en', limit=1)
    articles = client.articles.fetch_articles(params)
    headline = articles.articles[0].title
    sentiment = get_sentiment(headline)
    
    print(f"News: {headline}")
    print(f"Sentiment: {sentiment:+.4f}")
    
    # Get predictions
    data = prepare_data(ticker, years=2)
    scaled = data['scaled']
    raw = data['raw_ohlcv']
    dates = data['dates_features']
    
    train_end = int(len(scaled) * 0.70)
    
    print(f"\n| Day | Date       | Actual Price | Predicted | With Sentiment |")
    print(f"|----|------------|-------------|-----------|----------------|")
    
    for i in range(train_end, train_end + 10):
        date = dates[i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        
        model = load_model(f'models/{ticker}_lstm_ohlcv_indicators_v4.h5')
        pred = model.predict(scaled[i - SEQ_LEN:i].reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
        
        pred_open = base * np.exp(pred[0])
        pred_close = base * np.exp(pred[1])
        
        # With sentiment
        pred_with_sent = base * np.exp(pred[1] + sentiment * 0.3)
        
        print(f"| {i-train_end+1:2d} | {str(date)[:10]} | {actual:11.2f} | {pred_close:10.2f} | {pred_with_sent:13.2f} |")

print("\n" + "="*80)
print("Day = which test day (1 = first test day, etc.)")
print("Actual Price = the real closing price at end of that day")
print("Predicted = LSTM prediction without sentiment")
print("With Sentiment = LSTM + news sentiment adjustment")
print("="*80)