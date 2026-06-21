# ============================================================
# data_pipeline.py
# OHLCV log-returns + Technical Indicators + Candlestick ratios
#
# Features (13 total):
#   [0] Open        daily log-return
#   [1] High        daily log-return
#   [2] Low         daily log-return
#   [3] Close       daily log-return
#   [4] Volume      daily log-return
#   [5] RSI_14      raw value (0-100)
#   [6] MACD        raw value
#   [7] BB_position raw value (0-1)
#   [8] gap         (open - prev_close) / prev_close
#   [9] range_pct   (high - low) / close
#   [10] body_pct   (close - open) / open
#   [11] upper_wick (high - max(open, close)) / close
#   [12] lower_wick (min(open, close) - low) / close
#
# Targets: Open daily log-return + Close daily log-return
# ============================================================
import numpy as np
import pandas as pd
import yfinance as yf
import time
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler

# Column indices (feature matrix columns)
# 0..4: OHLCV daily log-returns (Open/High/Low/Close/Volume)
# 5..7: Technical indicators (RSI_14, MACD, BB_position)
# 8..12: Candlestick ratios (gap, range_pct, body_pct, upper_wick, lower_wick)
OPEN_IDX   = 0
HIGH_IDX   = 1
LOW_IDX    = 2
CLOSE_IDX  = 3
VOLUME_IDX = 4
RSI_IDX    = 5
MACD_IDX   = 6
BB_IDX     = 7

GAP_IDX       = 8
RANGE_PCT_IDX = 9
BODY_PCT_IDX  = 10
UPPER_WICK_IDX= 11
LOWER_WICK_IDX= 12

N_FEATURES = 13
N_OUTPUTS  = 2
SEQ_LEN    = 60


def load_price(symbol, years=10):
    cache_dir = os.path.join(os.path.dirname(__file__), "cache_prices")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_{years}y.pkl")

    today = pd.Timestamp.today().normalize()
    # Refresh quickly so the app always uses the latest prices.
    stale_threshold = pd.Timedelta(days=1)
    need_download = True

    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        loaded_from_cache = not df.empty
        if loaded_from_cache:
            last_dt = pd.Timestamp(df.index.max()).normalize()
            need_download = last_dt < (today - stale_threshold)
    else:
        loaded_from_cache = False
        need_download = True

    if need_download:
        start_ts = pd.Timestamp.today() - pd.DateOffset(years=years)
        end_ts = today + pd.Timedelta(days=5)

        last_err = None
        for _ in range(4):
            try:
                df = yf.download(
                    symbol,
                    start=start_ts,
                    end=end_ts,
                    progress=False,
                    threads=False,
                )
                if df is not None and not df.empty:
                    break
                last_err = ValueError(f"Empty dataframe for {symbol}")
            except Exception as e:
                last_err = e
                df = None
            time.sleep(3)
        else:
            raise ValueError(f"yfinance download failed for {symbol}: {last_err}")

    if df.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[df.index.dayofweek < 5]
    holidays = USFederalHolidayCalendar().holidays(
        start=df.index.min(), end=df.index.max()
    )
    df = df[~df.index.isin(holidays)]
    df.dropna(inplace=True)

    # Save clean cache
    df.to_pickle(cache_path)
    return df


def compute_rsi(series, period=14):
    """
    RSI - Relative Strength Index (0-100)
    Why: Most used momentum indicator.
    Captures overbought/oversold conditions.
    LSTM learns mean-reversion patterns from it.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """
    MACD Histogram = MACD line - Signal line
    MACD line = EMA(12) - EMA(26)
    Signal line = 9-day EMA of MACD line
    Why: Captures trend direction and momentum. Histogram shows momentum strength.
    Positive = bullish momentum increasing, Negative = bearish momentum increasing.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # MACD Histogram


def compute_bb_position(series, period=20, std_dev=2):
    """
    Bollinger Band Position (0 to 1)
    0 = lower band (oversold), 1 = upper band (overbought)
    Why: Different from RSI - price-band based not momentum.
    Captures volatility regime and mean reversion.
    """
    ma       = series.rolling(period).mean()
    std      = series.rolling(period).std()
    upper    = ma + (std_dev * std)
    lower    = ma - (std_dev * std)
    position = (series - lower) / (upper - lower + 1e-10)
    return position.clip(0, 1)


def add_indicators(df):
    """Add RSI, MACD, BB_position to dataframe."""
    close            = df["Close"]
    df["RSI_14"]     = compute_rsi(close)
    df["MACD"]       = compute_macd(close)
    df["BB_position"]= compute_bb_position(close)
    df.dropna(inplace=True)
    return df


def build_feature_matrix(df):
    """
    Build feature matrix.
    OHLCV -> daily log-returns (stationary)
    Indicators -> raw values (RSI_14, MACD, BB_position)
    Candlestick ratios -> normalized body/wick/range/gap descriptors
    """
    raw_ohlcv   = df[["Open","High","Low","Close","Volume"]].values
    # Daily log-returns for all OHLCV columns.
    price_rets  = np.log(raw_ohlcv[1:] / (raw_ohlcv[:-1] + 1e-10))
    indicators  = df[["RSI_14","MACD","BB_position"]].values[1:]

    # Candlestick-derived ratios computed from real prices.
    # They are aligned with the same day index as `price_rets` (day t uses t vs t-1).
    open_t  = raw_ohlcv[1:, 0]
    high_t  = raw_ohlcv[1:, 1]
    low_t   = raw_ohlcv[1:, 2]
    close_t = raw_ohlcv[1:, 3]
    prev_close = raw_ohlcv[:-1, 3]

    gap = (open_t - prev_close) / (prev_close + 1e-10)
    range_pct = (high_t - low_t) / (close_t + 1e-10)
    body_pct = (close_t - open_t) / (open_t + 1e-10)
    upper_wick = (high_t - np.maximum(open_t, close_t)) / (close_t + 1e-10)
    lower_wick = (np.minimum(open_t, close_t) - low_t) / (close_t + 1e-10)

    candlestick = np.column_stack(
        [gap, range_pct, body_pct, upper_wick, lower_wick]
    )

    features = np.column_stack([price_rets, indicators, candlestick])
    return features, raw_ohlcv


def scale_features(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def inverse_transform_col(scaled_values, col_idx, scaler):
    """
    Correct inverse transform for a single column.
    Uses scaler internals directly - no dummy matrix distortion.
    result = scaled_value * std + mean
    """
    mean = scaler.mean_[col_idx]
    std  = scaler.scale_[col_idx]
    return np.array(scaled_values) * std + mean


def create_sequences(scaled_data, seq_len=SEQ_LEN):
    """
    X shape: (N, seq_len, 8)
    y shape: (N, 2) -> [Open return, Close return]
    """
    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, :])
        y.append([scaled_data[i, OPEN_IDX],
                  scaled_data[i, CLOSE_IDX]])
    return np.array(X), np.array(y)


def train_test_split_temporal(X, y, train_ratio=0.7):
    """Temporal split - 70% train, 15% validation, 15% test."""
    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * 0.85)
    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]


def walk_forward_split(X, y, n_folds=7, test_size=30):
    """
    Walk-forward validation with 7 folds.
    Each fold: train on past data, test on next 30 days.
    """
    n = len(X)
    # Start from 70% mark
    train_start = int(n * 0.70)
    
    folds = []
    for i in range(n_folds):
        # Test period: test_size days starting from train_start + i*test_size
        test_start = train_start + i * test_size
        test_end = min(test_start + test_size, n)
        
        if test_end >= n:
            break
        
        # Train: from beginning to test_start
        X_train_fold = X[:test_start]
        y_train_fold = y[:test_start]
        
        # Test: test_start to test_end
        X_test_fold = X[test_start:test_end]
        y_test_fold = y[test_start:test_end]
        
        folds.append({
            'X_train': X_train_fold,
            'y_train': y_train_fold,
            'X_test': X_test_fold,
            'y_test': y_test_fold,
            'test_start_index': test_start,
            'test_end_index': test_end,
            'fold_num': i + 1
        })
    
    return folds


def prepare_data(symbol, years=10):
    """Full pipeline - call once per symbol."""
    cache_dir = os.path.join(os.path.dirname(__file__), "cache_features")
    os.makedirs(cache_dir, exist_ok=True)
    # Bump this whenever feature construction changes (shape mismatch protection).
    features_version = 5
    cache_path = os.path.join(
        cache_dir, f"{symbol}_{years}y_prepared_v{features_version}.pkl"
    )

    today = pd.Timestamp.today().normalize()
    stale_threshold = pd.Timedelta(days=1)
    if os.path.exists(cache_path):
        try:
            cached = pd.read_pickle(cache_path)
            df_cached = cached.get("df") if isinstance(cached, dict) else None
            if df_cached is not None and not df_cached.empty:
                last_dt = pd.Timestamp(df_cached.index.max()).normalize()
                if last_dt >= (today - stale_threshold):
                    print(f"[data] prepare_data cache HIT for {symbol} -> {cache_path}")
                    return cached
        except Exception:
            # Fall back to recompute.
            pass

    print(f"[data] prepare_data cache MISS for {symbol} -> recomputing")

    df                           = load_price(symbol, years)
    df                           = add_indicators(df)
    features, raw_ohlcv          = build_feature_matrix(df)
    # features are built from returns df.iloc[1:], so dates must align to df index starting at 2nd row
    dates_features               = df.index[1:]
    scaled, scaler               = scale_features(features)
    X, y                         = create_sequences(scaled)
    X_orig, _                    = create_sequences(raw_ohlcv)  # Original (unscaled) for baseline
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_temporal(X, y)
    result = {
        "df": df, "features": features, "raw_ohlcv": raw_ohlcv,
        "scaled": scaled, "scaler": scaler,
        "X": X, "y": y,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_orig": X_orig,
        "dates_features": dates_features,
    }

    # Cache full preprocessing to make symbol switching fast.
    try:
        pd.to_pickle(result, cache_path)
    except Exception:
        pass

    return result
