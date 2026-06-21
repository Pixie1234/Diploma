#!/usr/bin/env python3
"""
Create Base LSTM model (OHLCV only - no indicators)
Then run full evaluation
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("="*70)
print("STEP 1: Creating Base LSTM models (OHLCV only - 5 features)")
print("="*70)

N_FEATURES_BASE = 5  # OHLCV only
SEQ_LEN = 60

# Download stock data
stocks = ['AAPL', 'JPM']

for ticker in stocks:
    print(f"\n--- {ticker} ---")
    
    # Download data
    df = yf.download(ticker, period='2y', auto_adjust=False)
    df = df.dropna()
    
    # Create returns
    returns = np.log(df / df.shift(1)).dropna()
    
    # OHLCV only (5 features), but target is Close return (1 feature)
    # Actually target should be just Close return
    target_cols = ['Close']  # Just predict Close
    target_data = returns[target_cols].values
    
    # Scale separately
    scaler_feat = StandardScaler()
    scaler_tgt = StandardScaler()
    
    scaled_feat = scaler_feat.fit_transform(feature_data)
    scaled_tgt = scaler_tgt.fit_transform(target_data)
    
    # Sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled_feat)):
        X.append(scaled_feat[i-SEQ_LEN:i])  # All 5 features
        y.append(scaled_tgt[i])  # Close only
    X, y = np.array(X), np.array(y)
    
    print('X shape:', X.shape, 'y shape:', y.shape)
    
    # Split
    train_idx = int(len(X) * 0.65)
    X_train, y_train = X[:train_idx], y[:train_idx]
    
    # Build model (same architecture as LSTM+Tech)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, N_FEATURES_BASE)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(2)  # Open + Close
    ])
    model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
    
    # Train
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)
    
    # Save
    path = 'models/{}_lstm_v6.h5'.format(ticker)
    model.save(path)
    print(f"Saved: {path}")

print("\n" + "="*70)
print("STEP 2: Running Complete Evaluation")
print("="*70)

# Now evaluate all three models
from data_pipeline import prepare_data, SEQ_LEN, N_FEATURES
from lstm_model import load_model

# Get sentiment function
from finlight_client import FinlightApi, ApiConfig
from finlight_client.models import GetArticlesParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

API_KEY = 'sk_c3944c5fd6706ce5293517e8187d05f5c275d932389841cd156ed97ecdfc04ba'
DEVICE = 'cpu'
fin_tok = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
fin_mod = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone').to(DEVICE)
fin_mod.eval()
roberta = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest', device=-1)

def get_sentiment(ticker):
    client = FinlightApi(ApiConfig(api_key=API_KEY))
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

for ticker in stocks:
    print(f"\n{'='*40}")
    print(f"{ticker}")
    print(f"{'='*40}")
    
    sentiment = get_sentiment(ticker)
    print(f"Sentiment: {sentiment:+.4f}")
    
    # Data
    data = prepare_data(ticker, years=2)
    scaled = data['scaled']
    raw = data['raw_ohlcv']
    train_end = int(len(scaled) * 0.65)
    
    # Load models
    try:
        base_model = load_model('models/{}_lstm_v6.h5'.format(ticker))
    except:
        base_model = None
    
    tech_model = load_model('models/{}_lstm_ohlcv_indicators_v4.h5'.format(ticker))
    
    # Test
    base_preds = []
    tech_preds = []
    sent_preds = []
    actuals = []
    
    for i in range(train_end, train_end + 30):
        if i >= len(scaled) - 1:
            break
        
        last_seq = scaled[i - SEQ_LEN:i]
        actual = raw[i + 1, 3]
        base = raw[i, 3]
        
        # Base LSTM
        if base_model:
            pred = base_model.predict(last_seq.reshape(1, SEQ_LEN, 5), verbose=0)[0]
            base_preds.append(base * np.exp(pred[1]))
        
        # LSTM+Tech
        pred = tech_model.predict(last_seq.reshape(1, SEQ_LEN, N_FEATURES), verbose=0)[0]
        tech_preds.append(base * np.exp(pred[1]))
        sent_preds.append(base * np.exp(pred[1] + sentiment * 0.3))
        
        actuals.append(actual)
    
    # Accuracy
    actual_dir = np.array(actuals[1:]) > np.array(actuals[:-1])
    
    print(f"\n{'Model':<30} {'Dir Acc':>10}")
    print(f"{'-'*40}")
    
    if base_preds:
        base_dir = np.array(base_preds[1:]) > np.array(base_preds[:-1])
        print(f"{'LSTM (base)':<30} {np.mean(base_dir == actual_dir)*100:>10.1f}%")
    
    tech_dir = np.array(tech_preds[1:]) > np.array(tech_preds[:-1])
    print(f"{'LSTM + Tech Indicators':<30} {np.mean(tech_dir == actual_dir)*100:>10.1f}%")
    
    sent_dir = np.array(sent_preds[1:]) > np.array(sent_preds[:-1])
    print(f"{'Proposed (LSTM+Tech+Sent)':<30} {np.mean(sent_dir == actual_dir)*100:>10.1f}%")