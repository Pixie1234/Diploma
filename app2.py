# ============================================================
# app.py - Streamlit
# Features: OHLCV + RSI + MACD + BB_position (8 features)
# Predicts: Open + Close (dual output)
# ============================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DEVICE = torch.device("cpu")

_NY = ZoneInfo("America/New_York")
_UTC_MIN = datetime.min.replace(tzinfo=timezone.utc)

from calendar_dates import (
    get_next_trading_days, get_last_trading_day
)
from data_pipeline import (
    prepare_data, N_FEATURES, N_OUTPUTS, SEQ_LEN,
    OPEN_IDX, CLOSE_IDX, inverse_transform_col
)
from lstm_model import load_or_train, forecast_ohlcv
from informer_model import (
    load_or_train_informer,
    predict_informer,
    forecast_ohlcv_informer,
)

from news_sentiment_helpers import (
    normalize_article_text,
    company_tokens_from_name,
    finlight_pub_dt,
    is_finlight_relevant,
    alpha_pub_dt,
    is_alpha_vantage_relevant,
)
from sentiment2 import (
    load_nlp, finbert_sentiment, roberta_sentiment,
    fuse_sentiment, compute_total_bias, apply_sentiment_fusion,
    IMPACT_CLIP_POS, IMPACT_CLIP_NEG,
    IMPACT_CLIP_POS_CLOSE, IMPACT_CLIP_NEG_CLOSE,
    CONFLICT_PENALTY,
)
from evaluation import (
    evaluate_predictions, baseline_comparison,
    mcnemar_significance, ablation_summary
)
from sp500 import load_sp500

# ============================================================
# PAGE CONFIG
# ============================================================
try:
    st.set_page_config(
        page_title="Fused Market Predictor",
        layout="centered"
    )
except Exception:
    pass  # Already set in cached run

st.title("Fused Market Predictor")
st.caption(
    "LSTM + FinBERT + RoBERTa | Features: OHLCV + RSI + MACD + Bollinger | Predicts: Open & Close"
)

# Ensure news text wraps even if a source summary contains long
# unbroken tokens (e.g. missing spaces around numbers/words).
st.markdown(
    """
<style>
  .news-summary {
    overflow-wrap: anywhere;
    word-break: break-word;
    white-space: pre-wrap;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
@st.cache_data
def load_sp500_cached():
    return load_sp500()

df_sp500        = load_sp500_cached()

if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = None

company = st.sidebar.selectbox(
    "S&P 500 Company", df_sp500["Security"]
)
symbol = df_sp500[
    df_sp500["Security"] == company
].iloc[0]["Symbol"]

if st.session_state.last_symbol != symbol:
    st.session_state.last_symbol = symbol
    st.session_state.news_fetched = False

st.sidebar.success(f"{company} ({symbol})")

days_to_predict = st.sidebar.number_input(
    label="Forecast horizon (trading days)",
    min_value=5, max_value=90,
    value=30, step=1, format="%d"
)

model_mode = "LSTM vs Informer (Close)"

# New model version to avoid loading incompatible architectures.
model_path = f"models/{symbol}_lstm_ohlcv_indicators_v17.h5"
informer_model_path = f"models/{symbol}_informer_ohlcv_indicators_v2.pt"
informer_model = None

# ============================================================
# LOAD & PREPARE DATA
# ============================================================
@st.cache_resource
def get_data(symbol, _refresh_date: str):
    # _refresh_date is only used to bust cache once per day.
    # NOTE: keep args minimal; refresh date should only be used for cache invalidation.
    return prepare_data(symbol)

with st.spinner(f"Loading data for {symbol}..."):
    # Include feature-version in the cache-busting key.
    cache_bust_key = f"{pd.Timestamp.today().date()}_v4"
    ctx = get_data(symbol, cache_bust_key)

# Cache sanity check: ensure X_train matches current N_FEATURES.
# If not, recompute directly to avoid stale Streamlit cache.
if ctx.get("X_train") is not None and ctx["X_train"].shape[-1] != N_FEATURES:
    st.warning(
        "Stale preprocessing cache detected (feature dimension mismatch). "
        "Recomputing prepare_data(symbol) directly..."
    )
    ctx = prepare_data(symbol)

data     = ctx["df"]
if data.empty:
    st.error("No price data available.")
    st.stop()

st.sidebar.info(
    f"Data: {data.index[0].date()} to {data.index[-1].date()}\n"
    f"({len(data)} trading days)\n"
    f"Features: {N_FEATURES} (OHLCV+RSI+MACD+BB+candlestick ratios)"
)

# ============================================================
# LOAD / TRAIN MODEL
# ============================================================
@st.cache_resource
def get_model(symbol, path, n_features: int, data_version: str):
    # Cache only by (symbol, path) to avoid hashing large arrays.
    # Must use same feature-bust key as the main get_data call.
    refresh_date = f"{pd.Timestamp.today().date()}_v4"
    ctx = get_data(symbol, refresh_date)
    return load_or_train(
        ctx["X_train"], ctx["y_train"],
        ctx["X_val"], ctx["y_val"],
        path,
    )

with st.spinner("Loading model..."):
    model, was_loaded = get_model(
        symbol,
        model_path,
        N_FEATURES,
        data_version=f"features_v4_target_intraday",
    )

@st.cache_resource
def get_informer(symbol, path, data_version: str):
    # Must use same feature-bust key as the main get_data call.
    refresh_date = f"{pd.Timestamp.today().date()}_v4"
    ctx = get_data(symbol, refresh_date)
    # Always ensure the Informer model exists: load if present, otherwise train.
    if ctx.get("X_train") is not None and ctx["X_train"].shape[-1] != N_FEATURES:
        # Extra safety: stale Streamlit caches can survive code edits.
        ctx = prepare_data(symbol)
    return load_or_train_informer(
        ctx["X_train"], ctx["y_train"],
        ctx["X_val"], ctx["y_val"],
        path,
    )

with st.spinner("Loading Informer model..."):
    informer_out = get_informer(
        symbol,
        informer_model_path,
        data_version=f"features_v4_target_intraday",
    )
    informer_model, informer_was_loaded = informer_out

model_status = "Loaded saved model" if was_loaded else "Model trained"
st.success(
    f"✓ {model_status} | Input: {N_FEATURES} features | Output: {N_OUTPUTS} (Open + Close)"
)

DECAY_RATE = 10.0
ARTICLE_SCALE = 0.10
TOTAL_SCALE = 0.20
# Investors don't need internal fusion parameters.
st.caption(
    f"Fusion params: decay={DECAY_RATE}, "
    f"article_scale={ARTICLE_SCALE}, total_scale={TOTAL_SCALE}"
)

# ============================================================
# MODEL EVALUATION
# ============================================================
st.subheader("📊 Model Evaluation on Test Set")

# Debug/diagnostics are disabled by default (keep UI clean).
show_debug = False

# Close-only calibration to improve directional metrics.
# This is deterministic post-processing (train-only threshold tuning + variance scaling)
# and does not change the model weights.
enable_close_calibration = True

# Used for converting log-return error into approximate dollar error.
last_open_price = float(ctx["raw_ohlcv"][-1, OPEN_IDX])
last_close_price = float(ctx["raw_ohlcv"][-1, CLOSE_IDX])

# Diagnostics requested: standard deviation of scaled targets.
# If Close std were extremely small, Direction would tend to look random.
if show_debug:
    with st.expander("Diagnostics: target std (scaled, Train)"):
        y_open_train_scaled = ctx["y_train"][:, 0]
        y_close_train_scaled = ctx["y_train"][:, 1]
        st.write(
            "np.std(ctx['y_train'][:,0]) (Open): "
            f"{float(np.std(y_open_train_scaled)):.6f}"
        )
        st.write(
            "np.std(ctx['y_train'][:,1]) (Close): "
            f"{float(np.std(y_close_train_scaled)):.6f}"
        )


# Hard compatibility check before prediction to avoid predict-time retry loops.
def _model_expected_n_features(m):
    try:
        inp_shape = getattr(m, "input_shape", None)
        if inp_shape is not None and isinstance(inp_shape, (tuple, list)):
            if len(inp_shape) >= 3:
                return int(inp_shape[-1])
    except Exception:
        pass
    try:
        if getattr(m, "inputs", None):
            t0 = m.inputs[0]
            if t0.shape is not None and len(t0.shape) >= 3:
                return int(t0.shape[-1])
    except Exception:
        pass
    return None


X_n_features = int(ctx["X_test"].shape[-1])
model_expected_features = _model_expected_n_features(model)

if model_expected_features is not None and model_expected_features != X_n_features:
    # Delete stale model so load_or_train cannot keep loading an incompatible file.
    if model_path and os.path.exists(model_path):
        try:
            os.remove(model_path)
        except Exception:
            pass

    model, was_loaded = load_or_train(
        ctx["X_train"], ctx["y_train"],
        ctx["X_val"], ctx["y_val"],
        model_path,
    )

    model_expected_features_after = _model_expected_n_features(model)
    if model_expected_features_after is not None and model_expected_features_after != X_n_features:
        with st.expander("Model feature mismatch: retrain once"):
            st.error(
                "Model expects different input feature dimension even after retrain/reload. "
                f"Before={model_expected_features}, after={model_expected_features_after}, data={X_n_features}."
            )
            st.write(f"Retrain/Reload complete. was_loaded={was_loaded}")


# Predict once (compatibility is checked above).
pred_out = model.predict(ctx["X_test"], verbose=0)
if isinstance(pred_out, (list, tuple)):
    # [open_return, close_return] each shaped (N, 1)
    y_pred_both = np.hstack([pred_out[0], pred_out[1]])
else:
    y_pred_both = pred_out

# Optional post-processing calibration.
y_pred_both_cal = y_pred_both
if show_debug:
    use_close_calibration = st.checkbox(
        "Enable Close calibration (post-processing)",
        value=False,
    )

    if use_close_calibration:
        # Gentle variance-only inflation based on train-set real log-return std.
        with st.expander("Calibration: gentle std-only (train)"):
            pred_train_out = model.predict(ctx["X_train"], verbose=0)
            if isinstance(pred_train_out, (list, tuple)):
                y_pred_train_both = np.hstack([pred_train_out[0], pred_train_out[1]])
            else:
                y_pred_train_both = pred_train_out
            y_true_train_close_real = inverse_transform_col(
                ctx["y_train"][:, 1], CLOSE_IDX, ctx["scaler"]
            )
            y_pred_train_close_real = inverse_transform_col(
                y_pred_train_both[:, 1], CLOSE_IDX, ctx["scaler"]
            )

            true_std = float(np.std(y_true_train_close_real))
            pred_std = float(np.std(y_pred_train_close_real))

            beta = 0.5
            max_r = 2.0
            if pred_std > 0:
                r = (true_std / pred_std) ** beta
                r = float(np.clip(r, 0.0, max_r))
            else:
                r = 1.0

            y_pred_test_close_real = inverse_transform_col(
                y_pred_both[:, 1], CLOSE_IDX, ctx["scaler"]
            )
            y_pred_test_close_real_cal = y_pred_test_close_real * r

            st.write(
                f"Train true std={true_std:.6f}, pred std={pred_std:.6f}, r={r:.4f}"
            )

            mean_close = float(ctx["scaler"].mean_[CLOSE_IDX])
            std_close = float(ctx["scaler"].scale_[CLOSE_IDX])
            y_pred_both_cal = y_pred_both.copy()
            y_pred_both_cal[:, 1] = (y_pred_test_close_real_cal - mean_close) / std_close

    # ── Direction threshold tuning (train) for Close ──
    if enable_close_calibration:
        # Predict on train for threshold selection.
        pred_train_out = model.predict(ctx["X_train"], verbose=0)
        if isinstance(pred_train_out, (list, tuple)):
            y_pred_train_both = np.hstack([pred_train_out[0], pred_train_out[1]])
        else:
            y_pred_train_both = pred_train_out

        y_true_train_close_real = inverse_transform_col(
            ctx["y_train"][:, 1], CLOSE_IDX, ctx["scaler"]
        )
        y_pred_train_close_real = inverse_transform_col(
            y_pred_train_both[:, 1], CLOSE_IDX, ctx["scaler"]
        )

        # Candidate thresholds from prediction distribution.
        qs = np.linspace(0.05, 0.95, 19)
        candidate_ts = np.quantile(y_pred_train_close_real, qs)

        best_t = 0.0
        best_acc = -1.0
        y_true_dir = (y_true_train_close_real > 0)
        for t in candidate_ts:
            acc = float(np.mean((y_pred_train_close_real > t) == y_true_dir))
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)

        # Convert subtracting threshold in real space to scaled-space adjustment:
        # scaled = (real - mean) / std  => scaled_adj = scaled - t/std
        scale_std_close = float(ctx["scaler"].scale_[CLOSE_IDX])
        y_pred_both_cal = y_pred_both_cal.copy()
        y_pred_both_cal[:, 1] = y_pred_both_cal[:, 1] - (best_t / scale_std_close)

    # ── Fix 4: variance scaling sanity check (post-processing only) ──
    if enable_close_calibration:
        train_std_close = float(np.std(ctx["y_train"][:, 1]))
        pred_std_close = float(np.std(y_pred_both_cal[:, 1]))
        if pred_std_close > 0:
            scale_close = train_std_close / pred_std_close
            # Clamp to avoid exploding outputs if something goes wrong.
            scale_close = float(np.clip(scale_close, 0.2, 5.0))
            y_pred_both_cal[:, 1] = y_pred_both_cal[:, 1] * scale_close

# (removed) automatic variance scaling for Close

# Evaluate Open prediction
m_open, df_open, yt_open, yp_open = evaluate_predictions(
    ctx["y_test"][:, 0], y_pred_both[:, 0],
    ctx["scaler"], OPEN_IDX, "Open"
)

# Evaluate Close prediction
m_close, df_close, yt_close, yp_close = evaluate_predictions(
    ctx["y_test"][:, 1], y_pred_both_cal[:, 1],
    ctx["scaler"], CLOSE_IDX, "Close"
)

# Faza 2 (variance check) shown only in debug mode.
if show_debug:
    st.write("Faza 2: variance check (Close)")
    st.write(
        f"np.std(ctx['y_train'][:,0]) (Open target): {float(np.std(ctx['y_train'][:,0])):.6f}"
    )
    st.write(
        f"np.std(ctx['y_train'][:,1]) (Close target): {float(np.std(ctx['y_train'][:,1])):.6f}"
    )
    st.write(
        f"np.std(y_pred_both[:,0]) (Open pred, scaled): {float(np.std(y_pred_both[:,0])):.6f}"
    )
    st.write(
        f"np.std(y_pred_both[:,1]) (Close pred, scaled): {float(np.std(y_pred_both[:,1])):.6f}"
    )
    with st.expander("Faza 2: variance check (Close)", expanded=True):
        st.write(
            f"np.std(y_pred_both_cal[:,1]) (Close pred calibrated, scaled): {float(np.std(y_pred_both_cal[:,1])):.6f}"
        )

# Display metrics side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Open Price Prediction")
    st.write(f"**RMSE:** {m_open['RMSE']:.6f}")
    st.write(f"**MAE:** {m_open['MAE']:.6f}")
    st.write(f"**Direction:** {m_open['Directional Accuracy']:.2%}")
    st.write(f"**F1:** {m_open['F1 Score']:.4f}")

    if m_open["Directional Accuracy"] > 0.60:
        st.success(f"✓ {m_open['Directional Accuracy']:.2%} — beats 60% target")
    elif m_open["Directional Accuracy"] > 0.52:
        st.warning(f"⚠ {m_open['Directional Accuracy']:.2%} — beats random")
    else:
        st.error(f"✗ {m_open['Directional Accuracy']:.2%} — at or below random")

with col2:
    st.subheader("Close Price Prediction")
    st.write(f"**RMSE:** {m_close['RMSE']:.6f}")
    st.write(f"**MAE:** {m_close['MAE']:.6f}")
    st.write(f"**Direction:** {m_close['Directional Accuracy']:.2%}")
    st.write(f"**F1:** {m_close['F1 Score']:.4f}")

    if show_debug:
        # Diagnostics: check whether Close target construction/scaling
        # could explain a near-random directional accuracy.
        with st.expander("Diagnostics: Close target stats (Test)"):
            y_close_scaled = ctx["y_test"][:, 1]
            y_close_real = inverse_transform_col(y_close_scaled, CLOSE_IDX, ctx["scaler"])
            st.write(f"CLOSE_IDX used for Close target: {CLOSE_IDX}")
            st.write(
                "Scaled Close target: mean/std (should be ~0/1 for StandardScaler): "
                f"{float(np.mean(y_close_scaled)):.6f} / {float(np.std(y_close_scaled)):.6f}"
            )
            st.write(
                "Scaler Close feature stats (mean/std): "
                f"{float(ctx['scaler'].mean_[CLOSE_IDX]):.6f} / {float(ctx['scaler'].scale_[CLOSE_IDX]):.6f}"
            )
            st.write(
                "Real (inverse) Close log-return stats: "
                f"mean/std={float(np.mean(y_close_real)):.6f}/{float(np.std(y_close_real)):.6f}"
            )
            st.write(
                "Positive ratio (y_true_close_real > 0): "
                f"{float(np.mean(y_close_real > 0)):.2%}"
            )

            st.write(
                "Predicted Close log-return stats (real): "
                f"mean/std={float(np.mean(yp_close)):.6f}/{float(np.std(yp_close)):.6f}"
            )
            st.write(
                "Positive ratio (y_pred_close_real > 0): "
                f"{float(np.mean(yp_close > 0)):.2%}"
            )
            corr = float(np.corrcoef(y_close_real, yp_close)[0, 1])
            st.write(f"Correlation (true vs predicted, Close): {corr:.4f}")

    if m_close["Directional Accuracy"] > 0.60:
        st.success(f"✓ {m_close['Directional Accuracy']:.2%} — beats 60% target")
    elif m_close["Directional Accuracy"] > 0.50:
        st.warning(f"⚠ {m_close['Directional Accuracy']:.2%} — beats random")
    else:
        st.error(f"✗ {m_close['Directional Accuracy']:.2%} — at or below random")

# Full metrics tables
with st.expander("Full Metrics Detail"):
    keep_metrics = ["Directional Accuracy", "Precision", "Recall", "MAE"]
    st.dataframe(
        df_open[df_open["Metric"].isin(keep_metrics)],
        use_container_width=True,
    )
    st.dataframe(
        df_close[df_close["Metric"].isin(keep_metrics)],
        use_container_width=True,
    )

if model_mode == "LSTM vs Informer (Close)":
    informer_y_pred_both = predict_informer(informer_model, ctx["X_test"])

    # ── Proposed (LSTM + technical indicators) + (Informer + technical indicators)
    # with sentiment fusion estimations on the test set.
    # NOTE: real historical news sentiment is unavailable for the full test period,
    # so we use a synthetic sentiment proxy derived from previous-day close returns.
    def _synthetic_total_bias_proxy_from_prev_close(test_close_scaled: np.ndarray) -> np.ndarray:
        close_real = inverse_transform_col(test_close_scaled, CLOSE_IDX, ctx["scaler"])
        prev = np.concatenate([[0.0], close_real[:-1]])
        BETA = 0.20
        SCALE_FACTOR = 20.0
        return np.clip(prev * SCALE_FACTOR, -BETA, BETA)

    total_bias_proxy = _synthetic_total_bias_proxy_from_prev_close(ctx["y_test"][:, 1])
    sentiment_dir = np.sign(total_bias_proxy)

    # LSTM predictions (scaled)
    y_pred_lstm_open_scaled = y_pred_both[:, 0].copy()
    y_pred_lstm_close_scaled = y_pred_both_cal[:, 1].copy()
    pred_lstm_close_real = inverse_transform_col(y_pred_lstm_close_scaled, CLOSE_IDX, ctx["scaler"])

    # Informer predictions (scaled)
    y_pred_inf_open_scaled = informer_y_pred_both[:, 0].copy()
    y_pred_inf_close_scaled = informer_y_pred_both[:, 1].copy()
    pred_inf_close_real = inverse_transform_col(y_pred_inf_close_scaled, CLOSE_IDX, ctx["scaler"])

    # Consistency factor: if sentiment dir and predicted momentum dir conflict, impact can be nullified.
    lstm_pred_dir = np.sign(pred_lstm_close_real)
    inf_pred_dir = np.sign(pred_inf_close_real)

    lstm_consistency = np.ones_like(total_bias_proxy, dtype=float)
    inf_consistency = np.ones_like(total_bias_proxy, dtype=float)

    conflict_lstm = (sentiment_dir != 0) & (lstm_pred_dir != 0) & (sentiment_dir != lstm_pred_dir)
    conflict_inf = (sentiment_dir != 0) & (inf_pred_dir != 0) & (sentiment_dir != inf_pred_dir)
    lstm_consistency[conflict_lstm] = float(CONFLICT_PENALTY)
    inf_consistency[conflict_inf] = float(CONFLICT_PENALTY)

    scale_std = ctx["scaler"].scale_

    impact_lstm_open_real = np.clip(total_bias_proxy * lstm_consistency, IMPACT_CLIP_NEG, IMPACT_CLIP_POS)
    impact_lstm_close_real = np.clip(
        total_bias_proxy * lstm_consistency,
        IMPACT_CLIP_NEG_CLOSE,
        IMPACT_CLIP_POS_CLOSE,
    )
    impact_inf_open_real = np.clip(total_bias_proxy * inf_consistency, IMPACT_CLIP_NEG, IMPACT_CLIP_POS)
    impact_inf_close_real = np.clip(
        total_bias_proxy * inf_consistency,
        IMPACT_CLIP_NEG_CLOSE,
        IMPACT_CLIP_POS_CLOSE,
    )

    # Convert impact (real log-return units) to scaled units.
    impact_lstm_open_scaled = impact_lstm_open_real / scale_std[OPEN_IDX]
    impact_lstm_close_scaled = impact_lstm_close_real / scale_std[CLOSE_IDX]
    impact_inf_open_scaled = impact_inf_open_real / scale_std[OPEN_IDX]
    impact_inf_close_scaled = impact_inf_close_real / scale_std[CLOSE_IDX]

    fused_lstm_open_scaled = y_pred_lstm_open_scaled + impact_lstm_open_scaled
    fused_lstm_close_scaled = y_pred_lstm_close_scaled + impact_lstm_close_scaled
    fused_inf_open_scaled = y_pred_inf_open_scaled + impact_inf_open_scaled
    fused_inf_close_scaled = y_pred_inf_close_scaled + impact_inf_close_scaled

    proposed_open_scaled = 0.5 * (fused_lstm_open_scaled + fused_inf_open_scaled)
    proposed_close_scaled = 0.5 * (fused_lstm_close_scaled + fused_inf_close_scaled)
    proposed_both_scaled = np.stack([proposed_open_scaled, proposed_close_scaled], axis=1)

    m_open_prop, _, yt_open_prop, yp_open_prop = evaluate_predictions(
        ctx["y_test"][:, 0], proposed_both_scaled[:, 0],
        ctx["scaler"], OPEN_IDX, "Open (Proposed)"
    )
    m_close_prop, _, yt_close_prop, yp_close_prop = evaluate_predictions(
        ctx["y_test"][:, 1], proposed_both_scaled[:, 1],
        ctx["scaler"], CLOSE_IDX, "Close (Proposed)"
    )

    # ── Ablation: Proposed without Informer ──────────────────────────────
    # If Informer is not contributing meaningfully, Proposed-by-fusion should
    # perform similarly to the LSTM-only fused branch.
    proposed_no_inf_open_scaled = fused_lstm_open_scaled
    proposed_no_inf_close_scaled = fused_lstm_close_scaled
    proposed_no_inf_both_scaled = np.stack(
        [proposed_no_inf_open_scaled, proposed_no_inf_close_scaled], axis=1
    )
    m_open_prop_no_inf, _, _, _ = evaluate_predictions(
        ctx["y_test"][:, 0], proposed_no_inf_both_scaled[:, 0],
        ctx["scaler"], OPEN_IDX, "Open (Proposed w/o Informer)"
    )
    m_close_prop_no_inf, _, _, _ = evaluate_predictions(
        ctx["y_test"][:, 1], proposed_no_inf_both_scaled[:, 1],
        ctx["scaler"], CLOSE_IDX, "Close (Proposed w/o Informer)"
    )

    m_close_inf, df_close_inf, yt_close_inf, yp_close_inf = evaluate_predictions(
        ctx["y_test"][:, 1], informer_y_pred_both[:, 1],
        ctx["scaler"], CLOSE_IDX, "Close (Informer)"
    )

    sig_close_between = mcnemar_significance(
        yt_close,
        yp_close,
        y_pred_baseline=yp_close_inf,
        label="Close (LSTM vs Informer)",
    )

    st.subheader("🧠 Informer vs LSTM (Close, 1-day ahead)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LSTM")
        st.metric("Win Rate", f"{m_close['Directional Accuracy']:.2%}")
        st.metric("Avg Daily Miss", f"{m_close['MAE'] * last_close_price:.4f}")
        st.metric("Uptrend Detection", f"{m_close['Recall']:.2%}")
    with col2:
        st.subheader("Informer")
        st.metric("Win Rate", f"{m_close_inf['Directional Accuracy']:.2%}")
        st.metric("Avg Daily Miss", f"{m_close_inf['MAE'] * last_close_price:.4f}")
        st.metric("Uptrend Detection", f"{m_close_inf['Recall']:.2%}")

    st.caption(
        "Directional difference significance compares LSTM vs Informer error patterns (McNemar)."
    )
    if sig_close_between["p_value"] < 0.05:
        st.success(
            f"✓ Directional difference is significant: p={sig_close_between['p_value']:.4f}"
        )
    else:
        st.warning(
            f"⚠ Not statistically significant: p={sig_close_between['p_value']:.4f}"
        )

    with st.expander("Informer Metrics Detail"):
        keep_metrics = ["Directional Accuracy", "Precision", "Recall", "MAE"]
        st.dataframe(
            df_close_inf[df_close_inf["Metric"].isin(keep_metrics)],
            use_container_width=True,
        )

    st.subheader("🚀 Proposed (LSTM + Informer + Sentiment, 1-day ahead)")
    st.caption(
        "⚠ Proposed backtest uses a synthetic sentiment proxy (prev-day close return) because real historical news sentiment is not available for the full test period."
    )
    colp1, colp2 = st.columns(2)
    with colp1:
        st.write("**Open**")
        st.write(f"RMSE: {m_open_prop['RMSE']:.6f}")
        st.write(f"MAE: {m_open_prop['MAE']:.6f}")
        st.write(f"Direction: {m_open_prop['Directional Accuracy']:.2%}")
        st.write(f"F1: {m_open_prop['F1 Score']:.4f}")
    with colp2:
        st.write("**Close**")
        st.write(f"RMSE: {m_close_prop['RMSE']:.6f}")
        st.write(f"MAE: {m_close_prop['MAE']:.6f}")
        st.write(f"Direction: {m_close_prop['Directional Accuracy']:.2%}")
        st.write(f"F1: {m_close_prop['F1 Score']:.4f}")

    with st.expander("Ablation: Proposed without Informer"):
        st.write("**Open (LSTM fused only)**")
        st.write(f"Direction: {m_open_prop_no_inf['Directional Accuracy']:.2%}")
        st.write("**Close (LSTM fused only)**")
        st.write(f"Direction: {m_close_prop_no_inf['Directional Accuracy']:.2%}")

# ── Baseline Comparison ───────────────────────────────────────
st.subheader("📊 Baseline Comparison")
st.caption(
    "Honest comparison against naive baselines. "
    "A good model must beat both."
)

comp_open, imp_open = baseline_comparison(yt_open, yp_open, "Open")
comp_close, imp_close = baseline_comparison(yt_close, yp_close, "Close")

col1, col2 = st.columns(2)
with col1:
    st.write("**Open**")
    st.dataframe(comp_open, use_container_width=True)
    is_dir_verdict = "direction" in str(imp_open.get("verdict", ""))
    if is_dir_verdict:
        delta = float(imp_open.get("direction_improvement", 0.0))
        delta_str = f"({delta:+.2f}pp Direction)"
        if delta > 0:
            st.success(f"✓ {imp_open['verdict']} {delta_str}")
        elif delta == 0:
            st.warning(f"⚠ {imp_open['verdict']} {delta_str}")
        else:
            st.error(f"✗ {imp_open['verdict']} {delta_str}")
    else:
        delta = float(imp_open["mse_improvement_pct"])
        if delta > 10:
            st.success(f"✓ {imp_open['verdict']} (+{delta:.1f}% MSE)")
        elif delta > 0:
            st.warning(f"⚠ {imp_open['verdict']} (+{delta:.1f}% MSE)")
        else:
            st.error(f"✗ {imp_open['verdict']} (+{delta:.1f}% MSE)")

with col2:
    st.write("**Close**")
    st.dataframe(comp_close, use_container_width=True)
    is_dir_verdict = "direction" in str(imp_close.get("verdict", ""))
    if is_dir_verdict:
        delta = float(imp_close.get("direction_improvement", 0.0))
        delta_str = f"({delta:+.2f}pp Direction)"
        if delta > 0:
            st.success(f"✓ {imp_close['verdict']} {delta_str}")
        elif delta == 0:
            st.warning(f"⚠ {imp_close['verdict']} {delta_str}")
        else:
            st.error(f"✗ {imp_close['verdict']} {delta_str}")
    else:
        delta = float(imp_close["mse_improvement_pct"])
        if delta > 10:
            st.success(f"✓ {imp_close['verdict']} (+{delta:.1f}% MSE)")
        elif delta > 0:
            st.warning(f"⚠ {imp_close['verdict']} (+{delta:.1f}% MSE)")
        else:
            st.error(f"✗ {imp_close['verdict']} (+{delta:.1f}% MSE)")

# ── Statistical Significance ──────────────────────────────────
st.subheader("📐 Statistical Significance (McNemar Test)")
st.caption(
    "Tests whether improvement over baseline is statistically "
    "significant or could be due to chance. "
    "p < 0.05 = significant for thesis."
)

sig_open = mcnemar_significance(yt_open, yp_open, label="Open")
sig_close = mcnemar_significance(yt_close, yp_close, label="Close")

col1, col2 = st.columns(2)
with col1:
    st.write(
        f"**Open** χ²={sig_open['chi2']}  p={sig_open['p_value']}"
    )
    if sig_open["significant"]:
        st.success(f"✓ {sig_open['conclusion']}")
    else:
        st.warning(f"⚠ {sig_open['conclusion']}")

with col2:
    st.write(
        f"**Close** χ²={sig_close['chi2']}  p={sig_close['p_value']}"
    )
    if sig_close["significant"]:
        st.success(f"✓ {sig_close['conclusion']}")
    else:
        st.warning(f"⚠ {sig_close['conclusion']}")

# ── Ablation Study ────────────────────────────────────────────
st.subheader("🔬 Ablation Study")
st.caption(
    "Shows contribution of each component. "
    "This table belongs directly in your thesis results chapter."
)
st.info(
    "Run the model with different feature sets to fill this table. "
    "Current run uses full feature set (OHLCV + RSI + MACD + BB)."
)

ablation_data = {
    "LSTM — OHLC only (baseline)": {"direction": 0.4959, "mae": 0.796519},
    "LSTM — OHLCV (+ Volume)": {"direction": 0.00, "mae": 0.00},
    "LSTM — OHLCV + RSI": {"direction": 0.00, "mae": 0.00},
    "LSTM — OHLCV + RSI + MACD": {"direction": 0.00, "mae": 0.00},
    "LSTM — OHLCV + RSI + MACD + BB (full)": {
        "direction": m_close["Directional Accuracy"],
        "mae": m_close["MAE"],
    },
}

st.dataframe(ablation_summary(ablation_data), use_container_width=True)
st.caption(
    "Fill in the empty rows by running the model "
    "with each configuration separately."
)
# ── Prediction Visualization ──────────────────────────────────
st.subheader("📈 Prediction Quality — Test Set")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Open: actual vs predicted
axes[0, 0].plot(yt_open[:100], label="Actual", alpha=0.8)
axes[0, 0].plot(yp_open[:100], label="Predicted", alpha=0.8)
axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.3)
axes[0, 0].set_title("Open Log Returns — Actual vs Predicted")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Close: actual vs predicted
axes[0, 1].plot(yt_close[:100], label="Actual", alpha=0.8)
axes[0, 1].plot(yp_close[:100], label="Predicted", alpha=0.8)
axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.3)
axes[0, 1].set_title("Close Log Returns — Actual vs Predicted")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Cumulative Open
axes[1, 0].plot(np.cumsum(yt_open), label="Actual Cumulative", lw=2)
axes[1, 0].plot(np.cumsum(yp_open), label="Predicted Cumulative", lw=2)
axes[1, 0].set_title("Cumulative Open Returns")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cumulative Close
axes[1, 1].plot(np.cumsum(yt_close), label="Actual Cumulative", lw=2)
axes[1, 1].plot(np.cumsum(yp_close), label="Predicted Cumulative", lw=2)
axes[1, 1].set_title("Cumulative Close Returns")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f"{symbol} — LSTM+Indicators Prediction Quality", fontsize=13)
plt.tight_layout()
st.pyplot(fig)

# ============================================================
# FORECAST
# ============================================================
st.header(f"Next {days_to_predict} Trading Days {symbol} price prediction")

last_real_day = get_last_trading_day(data.index[-1])
future_dates  = get_next_trading_days(last_real_day, days_to_predict)

weekend_check = future_dates[future_dates.dayofweek >= 5]
if len(weekend_check) > 0:
    st.error(f"Weekend dates in forecast: {weekend_check}")
else:
    st.success(
        f"✓ Forecast covers next {days_to_predict} market days (weekends and holidays excluded)"
    )

forecast = forecast_ohlcv(
    model,
    ctx["scaled"][-SEQ_LEN:],
    days_to_predict,
    ctx["scaler"],
    ctx["raw_ohlcv"]
)

informer_forecast = None
if model_mode == "LSTM vs Informer (Close)":
    informer_forecast = forecast_ohlcv_informer(
        informer_model,
        ctx["scaled"][-SEQ_LEN:],
        days_to_predict,
        ctx["scaler"],
        ctx["raw_ohlcv"],
    )

# Forecast table
st.subheader("Base Price Forecast (Price Patterns Only)")
forecast_df = pd.DataFrame({
    "Date":            future_dates.strftime("%Y-%m-%d (%A)"),
    "Base Open":       forecast["open_prices"],
    "Base Close":      forecast["close_prices"],
    **({
        "Informer Close": informer_forecast["close_prices"],
    } if informer_forecast is not None else {}),
    "Daily Range $":   [round(abs(o - c), 4)
                          for o, c in zip(
                              forecast["open_prices"],
                              forecast["close_prices"],
                          )]
})
st.dataframe(forecast_df, use_container_width=True)

# Forecast chart
fig_fc, ax = plt.subplots(figsize=(13, 5))
ax.plot(
    data.index[-90:], data["Close"].values[-90:],
    label="Historical Close", color="steelblue", lw=2
)
ax.plot(
    future_dates, forecast["open_prices"],
    label="Predicted Open", color="orange",
    marker="^", lw=2, linestyle="--"
)
ax.plot(
    future_dates, forecast["close_prices"],
    label="Predicted Close", color="green",
    marker="o", lw=2
)

if informer_forecast is not None:
    ax.plot(
        future_dates, informer_forecast["close_prices"],
        label="Informer Predicted Close", color="purple",
        marker="x", lw=2, linestyle=":"
    )
ax.fill_between(
    future_dates,
    forecast["open_prices"],
    forecast["close_prices"],
    alpha=0.15, color="gray", label="Open-Close range"
)
ax.axvline(
    x=last_real_day, color="red",
    linestyle="--", alpha=0.5, label="Forecast start"
)
ax.set_title(
    f"{symbol} — AI Base Forecast (Open & Close)"
)
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig_fc)

# ============================================================
# NLP MODELS
# ============================================================
@st.cache_resource
def get_nlp():
    return load_nlp()

# ============================================================
# NEWS + SENTIMENT FUSION
# ============================================================
st.header(f"📰 News Sentiment for {symbol}")

if "news_fetched" not in st.session_state:
    st.session_state.news_fetched = False

if st.button("Fetch News & Apply Sentiment", type="primary"):
    st.session_state.news_fetched = True

if st.session_state.news_fetched:
    try:
        # Load NLP models only when the user actually requests news sentiment.
        fin_tok, fin_mod, roberta = get_nlp()

        import concurrent.futures

        # Cache sentiment scoring per article text to avoid repeated FinBERT/RoBERTa inference.
        if "_sentiment_cache" not in st.session_state:
            st.session_state["_sentiment_cache"] = {}

        def score_article_text(text: str):
            cache = st.session_state["_sentiment_cache"]
            if text in cache:
                return cache[text]
            fin_l, fin_s = finbert_sentiment(text, fin_tok, fin_mod)
            rob_l, rob_s = roberta_sentiment(text, roberta)
            bias = fuse_sentiment(fin_l, fin_s, rob_l, rob_s)
            cache[text] = (fin_l, fin_s, rob_l, rob_s, bias)
            return cache[text]

        # ── Fetch Finlight + Alpha Vantage concurrently (network I/O only) ──
        from finlight_client import FinlightApi, ApiConfig
        from finlight_client.models import GetArticlesParams
        import requests

        finlight_key = st.secrets.get("FINLIGHT_API_KEY", "")  # type: ignore[attr-defined]
        if not finlight_key:
            finlight_key = os.environ.get("FINLIGHT_API_KEY", "")

        alpha_key = os.environ.get("ALPHA_VANTAGE_KEY", "")
        try:
            alpha_key = st.secrets.get("ALPHA_VANTAGE_KEY", alpha_key)  # type: ignore[attr-defined]
        except Exception:
            pass

        def fetch_finlight():
            if not finlight_key:
                return []
            client = FinlightApi(ApiConfig(api_key=finlight_key))
            params = GetArticlesParams(query=symbol, language='en', limit=10)
            articles = client.articles.fetch_articles(params)
            return (articles.articles[:10] if articles and articles.articles else [])

        def fetch_alpha_vantage():
            if not alpha_key:
                return {}
            url = "https://www.alphavantage.co/query"
            _params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": alpha_key,
                "limit": 10,
                "sort": "LATEST",
            }
            resp = requests.get(url, params=_params, timeout=15)
            return resp.json() if resp is not None else {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fin_future = ex.submit(fetch_finlight)
            alpha_future = ex.submit(fetch_alpha_vantage)
            articles_list = fin_future.result()
            data_av = alpha_future.result()

        if not finlight_key:
            st.warning("Finlight key missing (FINLIGHT_API_KEY). Using only Alpha Vantage news.")
        if not articles_list:
            st.warning("No Finlight news found. Trying Alpha Vantage...")

        feed_raw = []
        if not alpha_key:
            st.warning("Alpha Vantage key missing. Using only Finlight news.")
        else:
            if isinstance(data_av, dict) and "Information" in data_av:
                st.error("⚠ Alpha Vantage API limit: {}".format(data_av["Information"]))
            elif isinstance(data_av, dict) and "Note" in data_av:
                st.warning("⚠ Alpha Vantage note: {}".format(data_av["Note"]))
            else:
                feed_raw = (data_av.get("feed", []) or [])[:10]

        feed_raw_count = len(feed_raw)

        # ── STEP 1: collect + render ALL news (unified order) ──
        # We compute sentiment once per article, store results, then display
        # all items together sorted by publication datetime.
        biases     = []
        pub_times = []
        sentiments = []
        items = []

        # AlphaVantage/Finlight occasionally return strings with newlines/odd spacing.
        if "_url_works_cache" not in st.session_state:
            st.session_state["_url_works_cache"] = {}

        def _url_works_uncached(url: str) -> bool:
            import requests
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                # HEAD can be blocked by some hosts, so try HEAD then GET fallback.
                try:
                    resp = requests.head(
                        url,
                        allow_redirects=True,
                        timeout=3,
                        headers=headers,
                    )
                    if resp.status_code < 400:
                        return True
                except Exception:
                    pass

                resp = requests.get(
                    url,
                    stream=True,
                    allow_redirects=True,
                    timeout=3,
                    headers=headers,
                )
                return resp.status_code < 400
            except Exception:
                return False

        def _url_works(url: str) -> bool:
            cache = st.session_state["_url_works_cache"]
            if url in cache:
                return cache[url]
            ok = _url_works_uncached(url)
            cache[url] = ok
            return ok

        def _validate_urls(urls: list[str]) -> dict[str, bool]:
            cache = st.session_state["_url_works_cache"]
            uniq = sorted({u for u in urls if u})
            to_check = [u for u in uniq if u not in cache]
            if to_check:
                max_workers = min(8, len(to_check))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    results = list(ex.map(_url_works_uncached, to_check))
                for u, ok in zip(to_check, results):
                    cache[u] = ok
            return {u: cache.get(u, False) for u in uniq}

        company_tokens = company_tokens_from_name(company)
        symbol_upper = str(symbol).upper()

        # Remove generic company-name tokens to avoid over-filtering.
        generic_tokens = {
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
        company_tokens = [t for t in company_tokens if t not in generic_tokens]

        # ── Finlight items ──
        fin_items = [a for a in articles_list if is_finlight_relevant(a, symbol_upper=symbol_upper, company_tokens=company_tokens)]
        fin_items.sort(key=finlight_pub_dt, reverse=True)
        fin_items = fin_items[:5]

        def _build_finlight_items(_fin_items):
            prepared: list[dict] = []
            link_candidates: list[str] = []

            for art in _fin_items:
                title_raw = getattr(art, 'title', 'No title') or "No title"
                summary_raw = getattr(art, 'summary', None) or getattr(art, 'content', None) or ""
                title = normalize_article_text(str(title_raw), 140)
                summary = normalize_article_text(str(summary_raw), 200) if summary_raw else ""
                pub_dt = finlight_pub_dt(art)

                pub_date_simple = (
                    pub_dt.astimezone(_NY).date()
                    if pub_dt and pub_dt != datetime.min.replace(tzinfo=timezone.utc)
                    else None
                )

                # Finlight article link field can vary by SDK/version.
                link = (
                    getattr(art, 'url', None)
                    or getattr(art, 'link', None)
                    or getattr(art, 'source_url', None)
                    or getattr(art, 'document_url', None)
                    or ''
                )
                if not (isinstance(link, str) and link.strip().startswith(("http://", "https://"))):
                    link = ''
                else:
                    link = link.strip()
                    if " " in link:
                        link = ''

                if link:
                    link_candidates.append(link)

                prepared.append({
                    "pub_dt": pub_dt,
                    "pub_date_simple": pub_date_simple,
                    "title": title,
                    "summary": summary,
                    "link": link,
                })

            valid_map = _validate_urls(link_candidates) if link_candidates else {}

            built = []
            for p in prepared:
                title = p["title"]
                summary = p["summary"]
                link = p["link"]
                if link and not valid_map.get(link, False):
                    link = ''

                text = f"{title} {summary}".strip()
                fin_l, fin_s, rob_l, rob_s, bias = score_article_text(text)

                built.append({
                    "source": "Finlight",
                    "pub_dt": p["pub_dt"],
                    "pub_date_simple": p["pub_date_simple"],
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "fin_l": fin_l,
                    "fin_s": fin_s,
                    "rob_l": rob_l,
                    "rob_s": rob_s,
                    "bias": bias,
                })

            return built

        items.extend(_build_finlight_items(fin_items))

        # ── STEP 1b: Alpha Vantage items from the already fetched feed_raw ──
        if feed_raw:
            feed = [
                a for a in feed_raw
                if is_alpha_vantage_relevant(
                    a,
                    symbol_upper=symbol_upper,
                    company_tokens=company_tokens,
                )
            ]
            feed.sort(key=alpha_pub_dt, reverse=True)
            feed = feed[:5]

            def _build_alpha_vantage_items(_feed):
                prepared: list[dict] = []
                link_candidates: list[str] = []

                for art in _feed:
                    title = normalize_article_text(art.get("title", "No title") or "No title", 140)
                    summary = normalize_article_text(art.get("summary", "") or "", 600)
                    pub_dt = alpha_pub_dt(art)

                    pub_date_simple = (
                        pub_dt.astimezone(_NY).date()
                        if pub_dt and pub_dt != datetime.min.replace(tzinfo=timezone.utc)
                        else None
                    )

                    link = art.get("url", "") or ""
                    if not (isinstance(link, str) and link.strip().startswith(("http://", "https://"))):
                        link = ''
                    else:
                        link = link.strip()
                        if " " in link:
                            link = ''

                    if link:
                        link_candidates.append(link)

                    prepared.append({
                        "pub_dt": pub_dt,
                        "pub_date_simple": pub_date_simple,
                        "title": title,
                        "summary": summary,
                        "link": link,
                    })

                valid_map = _validate_urls(link_candidates) if link_candidates else {}

                built = []
                for p in prepared:
                    title = p["title"]
                    summary = p["summary"]
                    link = p["link"]
                    if link and not valid_map.get(link, False):
                        link = ''
                    if not link:
                        continue

                    text = f"{title} {summary}".strip()
                    fin_l, fin_s, rob_l, rob_s, bias = score_article_text(text)

                    built.append({
                        "source": "Alpha Vantage",
                        "pub_dt": p["pub_dt"],
                        "pub_date_simple": p["pub_date_simple"],
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "fin_l": fin_l,
                        "fin_s": fin_s,
                        "rob_l": rob_l,
                        "rob_s": rob_s,
                        "bias": bias,
                    })

                return built

            items.extend(_build_alpha_vantage_items(feed))

        # ── Unified render order ──
        if not items:
            st.warning("No relevant news articles found from any source.")
            st.session_state.news_fetched = False
            st.stop()

        excluded_titles = {
            "4 stocks to watch on thursday: ibm, mmm, kss, meta",
        }

        items.sort(
            key=lambda x: x.get("pub_date_simple") or datetime.min.date(),
            reverse=True,
        )

        for item in items:
            title_key = " ".join(str(item["title"]).split()).strip().lower()
            if title_key in excluded_titles or "4 stocks to watch on thursday" in title_key:
                continue

            st.markdown(f"### {item['title']}")
            pub_dt = item["pub_dt"]
            if pub_dt and pub_dt != datetime.min.replace(tzinfo=timezone.utc):
                pub_date_display = pub_dt.astimezone(
                    _NY
                ).date()
                st.caption(f"Published: {pub_date_display} | Source: {item['source']}")
            else:
                st.caption("Published: unknown | Source: " + item["source"])

            biases.append(item["bias"])
            pub_times.append(pub_dt if pub_dt != datetime.min.replace(tzinfo=timezone.utc) else None)
            sentiments.append({
                "Title": item["title"][:60] + "...",
                "FinBERT": f"{item['fin_l']} ({item['fin_s']:.2f})",
                "RoBERTa": f"{item['rob_l']} ({item['rob_s']:.2f})",
                "Source": item["source"],
            })

            col1, col2 = st.columns(2)
            with col1:
                if item["fin_l"] == "Bullish":
                    st.success(f"FinBERT Bullish ({item['fin_s']:.2f})")
                elif item["fin_l"] == "Bearish":
                    st.error(f"FinBERT Bearish ({item['fin_s']:.2f})")
                else:
                    st.info(f"FinBERT Neutral ({item['fin_s']:.2f})")
            with col2:
                rob_up = item["rob_l"].upper()
                if "POS" in rob_up:
                    st.success(f"RoBERTa {item['rob_l']} ({item['rob_s']:.2f})")
                elif "NEG" in rob_up:
                    st.error(f"RoBERTa {item['rob_l']} ({item['rob_s']:.2f})")
                else:
                    st.info(f"RoBERTa {item['rob_l']} ({item['rob_s']:.2f})")

            if item["summary"]:
                st.markdown(
                    f"<div class='news-summary'>{item['summary']}</div>",
                    unsafe_allow_html=True,
                )
            if item["link"]:
                st.markdown(f"[Read article]({item['link']})")
            st.divider()

        # ── STEP 2: aggregate with recency weighting ──
        total_bias = compute_total_bias(
            article_biases=biases,
            publish_times=pub_times,
        )

        signal = (
            "Bullish" if total_bias > 0.02 else
            "Bearish" if total_bias < -0.02 else
            "Neutral"
        )

        mood_emoji = "🟢" if signal == "Bullish" else ("🔴" if signal == "Bearish" else "🟡")
        st.subheader("Sentiment Summary")
        st.dataframe(pd.DataFrame(sentiments), use_container_width=True)
        st.info(
            f"Market Mood: {mood_emoji} {signal} | Based on {len(biases)} recent articles"
        )

        # ── STEP 3: compute recent volatility ──
        recent_log_returns = np.diff(np.log(data["Close"].values[-25:]))

        # ── STEP 4: apply improved fusion ──
        fused = apply_sentiment_fusion(
            total_bias=total_bias,
            open_returns=forecast["open_returns"],
            close_returns=forecast["close_returns"],
            last_open=float(ctx["raw_ohlcv"][-1, OPEN_IDX]),
            last_close=float(ctx["raw_ohlcv"][-1, CLOSE_IDX]),
            days_to_predict=days_to_predict,
            recent_returns=recent_log_returns,
            forecast_decay=10.0,
        )

        fused_open = fused["fused_open"]
        fused_close = fused["fused_close"]

        # If we also forecast with Informer, apply the same sentiment bias
        # to the Informer returns so the final forecast reflects the full model.
        fused_informer_open = None
        fused_informer_close = None
        proposed_open = None
        proposed_close = None
        if informer_forecast is not None:
            fused_inf = apply_sentiment_fusion(
                total_bias=total_bias,
                open_returns=informer_forecast["open_returns"],
                close_returns=informer_forecast["close_returns"],
                last_open=float(ctx["raw_ohlcv"][-1, OPEN_IDX]),
                last_close=float(ctx["raw_ohlcv"][-1, CLOSE_IDX]),
                days_to_predict=days_to_predict,
                recent_returns=recent_log_returns,
                forecast_decay=10.0,
            )
            fused_informer_open = fused_inf["fused_open"]
            fused_informer_close = fused_inf["fused_close"]

            # Proposed multimodel output: combine LSTM+Sentiment and Informer+Sentiment.
            proposed_open = [
                round((a + b) / 2.0, 4)
                for a, b in zip(fused_open, fused_informer_open)
            ]
            proposed_close = [
                round((a + b) / 2.0, 4)
                for a, b in zip(fused_close, fused_informer_close)
            ]

        # (Removed impact curve UI from the sentiment section.)

        # Fused forecast chart
        st.subheader(
            "Fused Forecast — LSTM + FinBERT + RoBERTa (with Informer proposed fusion)"
        )
        fig_f, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Open comparison
        axes[0].plot(
            data.index[-60:], data["Open"].values[-60:],
            label="Historical", color="steelblue", lw=2
        )
        axes[0].plot(
            future_dates, forecast["open_prices"],
            label="LSTM Only", color="orange",
            marker="^", lw=2, linestyle="--", alpha=0.7
        )
        axes[0].plot(
            future_dates, fused_open,
            label="LSTM + Sentiment", color="red",
            marker="^", lw=2
        )
        if informer_forecast is not None:
            axes[0].plot(
                future_dates, informer_forecast["open_prices"],
                label="Informer Only", color="teal",
                marker="^", lw=2, linestyle=":", alpha=0.9,
            )
        if fused_informer_open is not None:
            axes[0].plot(
                future_dates, fused_informer_open,
                label="Informer + Sentiment", color="purple",
                marker="^", lw=2, linestyle="-.",
            )
        if proposed_open is not None:
            axes[0].plot(
                future_dates, proposed_open,
                label="Proposed (LSTM+Informer+Sentiment)",
                color="black", marker="D", lw=2, linestyle="-",
            )
        axes[0].axvline(
            x=last_real_day, color="gray",
            linestyle="--", alpha=0.5
        )
        axes[0].set_title(f"{symbol} Open Forecast")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        plt.setp(
            axes[0].xaxis.get_majorticklabels(), rotation=45
        )

        # Close comparison
        axes[1].plot(
            data.index[-60:], data["Close"].values[-60:],
            label="Historical", color="steelblue", lw=2
        )
        axes[1].plot(
            future_dates, forecast["close_prices"],
            label="LSTM Only", color="green",
            marker="o", lw=2, linestyle="--", alpha=0.7
        )
        axes[1].plot(
            future_dates, fused_close,
            label="LSTM + Sentiment", color="darkgreen",
            marker="o", lw=2
        )
        if informer_forecast is not None:
            axes[1].plot(
                future_dates, informer_forecast["close_prices"],
                label="Informer Only", color="teal",
                marker="o", lw=2, linestyle=":", alpha=0.9,
            )
        if fused_informer_close is not None:
            axes[1].plot(
                future_dates, fused_informer_close,
                label="Informer + Sentiment", color="purple",
                marker="o", lw=2, linestyle="-.",
            )
        if proposed_close is not None:
            axes[1].plot(
                future_dates, proposed_close,
                label="Proposed (LSTM+Informer+Sentiment)",
                color="black", marker="s", lw=2, linestyle="-",
            )
        axes[1].axvline(
            x=last_real_day, color="gray",
            linestyle="--", alpha=0.5
        )
        axes[1].set_title(f"{symbol} Close Forecast")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.setp(
            axes[1].xaxis.get_majorticklabels(), rotation=45
        )

        plt.suptitle(f"{symbol} — LSTM vs Informer vs Proposed Forecast", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_f)

        # Comparison table
        table = {
            "Date": future_dates.strftime("%Y-%m-%d (%A)"),
            "LSTM Open": forecast["open_prices"],
            "Fused Open": fused_open,
            "Open Diff $": [
                round(f - b, 4)
                for b, f in zip(forecast["open_prices"], fused_open)
            ],
            "LSTM Close": forecast["close_prices"],
            "Fused Close": fused_close,
            "Close Diff $": [
                round(f - b, 4)
                for b, f in zip(forecast["close_prices"], fused_close)
            ],
        }

        if informer_forecast is not None:
            table["Informer Open"] = informer_forecast["open_prices"]
            table["Informer Close"] = informer_forecast["close_prices"]

        if fused_informer_open is not None:
            table["Informer + Sentiment Open"] = fused_informer_open
            table["Informer Open Diff $"] = [
                round(f - b, 4)
                for b, f in zip(informer_forecast["open_prices"], fused_informer_open)
            ]

        if fused_informer_close is not None:
            table["Informer + Sentiment Close"] = fused_informer_close
            table["Informer Close Diff $"] = [
                round(f - b, 4)
                for b, f in zip(informer_forecast["close_prices"], fused_informer_close)
            ]

        if proposed_open is not None:
            table["Proposed Open"] = proposed_open
            table["Proposed Open Diff $"] = [
                round(f - b, 4)
                for b, f in zip(forecast["open_prices"], proposed_open)
            ]

        if proposed_close is not None:
            table["Proposed Close"] = proposed_close
            table["Proposed Close Diff $"] = [
                round(f - b, 4)
                for b, f in zip(forecast["close_prices"], proposed_close)
            ]

        st.dataframe(pd.DataFrame(table), use_container_width=True)

    except Exception as e:
        st.error("Failed to fetch or process news")
        st.code(str(e))
        import traceback
        st.text(traceback.format_exc())
        traceback.print_exc()

# ============================================================
# EXPORT REPORT
# ============================================================
st.sidebar.divider()
if st.sidebar.button("📄 Download Performance Report"):
    report = f"""
STOCK PREDICTION EVALUATION REPORT
{symbol} - {company}
Features: OHLCV + RSI + MACD + BB_position (8 total)
Predicts: Open + Close (dual output)

OPEN PREDICTION
  Win Rate:           {m_open['Directional Accuracy']:.2%}
  Avg Daily Miss:     {m_open['MAE'] * last_open_price:.4f} $
  Uptrend Detection:  {m_open['Recall']:.2%}

CLOSE PREDICTION
  Win Rate:           {m_close['Directional Accuracy']:.2%}
  Avg Daily Miss:     {m_close['MAE'] * last_close_price:.4f} $
  Uptrend Detection:  {m_close['Recall']:.2%}
  ⚠ Close prediction is less reliable — use Open signal for decisions

BASELINE COMPARISON
  {'✓ AI model outperforms buy-and-hold by ' + f"{imp_close['mse_improvement_pct']:.1f}%" if imp_close['mse_improvement_pct'] > 0 else '⚠ Improvements are not consistent vs buy-and-hold'}

STATISTICAL SIGNIFICANCE
  {'✓ Verified with 99%+ confidence' if sig_open['p_value'] < 0.01 else '⚠ Not statistically verified — use with caution'}
  {'✓ Verified with 99%+ confidence' if sig_close['p_value'] < 0.01 else '⚠ Not statistically verified — use with caution'}

DATASET
  Training samples: {len(ctx['X_train'])}
  Validation samples: {len(ctx['X_val'])}
  Test samples:     {len(ctx['X_test'])}
  Sequence length:  {SEQ_LEN} trading days
  Features:         {N_FEATURES} (same for train/val/test)

DISCLAIMER: Research purposes only. Not financial advice.
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    st.sidebar.download_button(
        label="Download Report",
        data=report,
        file_name=f"{symbol}_thesis_evaluation.txt",
        mime="text/plain"
    )
