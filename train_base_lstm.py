#!/usr/bin/env python3
"""
Train Base LSTM (OHLCV only - 5 features, 1 output: Close)
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

print("="*60)
print("TRAINING BASE LSTM MODELS")
print("="*60)

SEQ_LEN = 60
N_FEATURES = 5  # OHLCV only

stocks = ['AAPL', 'JPM']

for ticker in stocks:
    print(f"\n--- {ticker} ---")
    
    # Download data
    df = yf.download(ticker, period='3y', auto_adjust=False)
    df = df.dropna()
    
    # Log returns
    returns = np.log(df / df.shift(1)).dropna()
    
    # Features: OHLCV (5 columns)
    feat_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    features = returns[feat_cols].values
    
    # Target: Close only (index 3)
    target = returns['Close'].values
    
    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(features)
    y = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Sequences
    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(X)):
        X_seq.append(X[i-SEQ_LEN:i])
        y_seq.append(y[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    
    # Split (65% train)
    train_end = int(len(X_seq) * 0.65)
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    
    print(f"Training data: {X_train.shape}")
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, N_FEATURES)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Just Close
    ])
    
    model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
    
    # Train
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Save
    path = 'models/{}_lstm_v6.h5'.format(ticker)
    model.save(path)
    print(f"Saved: {path}")

print("\n" + "="*60)
print("BASE LSTM MODELS CREATED!")
print("="*60)