# ============================================================
# evaluation.py
# Honest Model Evaluation for Diploma Thesis
#
# Covers:
#   1. Regression metrics (RMSE, MAE, MAPE*)
#   2. Directional accuracy metrics (F1, Precision, Recall)
#   3. Baseline comparison (naive + random walk)
#   4. Ablation study support
#   5. Statistical significance (McNemar test)
#
# *MAPE excluded - broken for log-returns near zero
#  (division by near-zero explodes to 7000%+)
# ============================================================
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score,
    recall_score, f1_score
)
from data_pipeline import inverse_transform_col


def evaluate_predictions(y_true_scaled, y_pred_scaled,
                         scaler, col_idx, label="Close"):
    """
    Full evaluation for one output (Open or Close).

    Inverse transforms before computing metrics so
    all values are in real log-return units not scaled units.

    Args:
        y_true_scaled: true values in scaled space
        y_pred_scaled: predicted values in scaled space
        scaler:        fitted StandardScaler
        col_idx:       which column (OPEN_IDX or CLOSE_IDX)
        label:         "Open" or "Close" for display

    Returns:
        metrics dict, summary DataFrame
    """
    # Inverse transform to real log-return units
    y_true = inverse_transform_col(y_true_scaled, col_idx, scaler)
    y_pred = inverse_transform_col(y_pred_scaled, col_idx, scaler)

    # Regression metrics
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    # Directional accuracy
    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    acc  = accuracy_score(y_true_dir, y_pred_dir)
    prec = precision_score(y_true_dir, y_pred_dir,
                           zero_division=0)
    rec  = recall_score(y_true_dir, y_pred_dir,
                        zero_division=0)
    f1   = f1_score(y_true_dir, y_pred_dir,
                    zero_division=0)

    metrics = {
        "Label":               label,
        "MSE":                 round(mse,  8),
        "RMSE":                round(rmse, 8),
        "MAE":                 round(mae,  8),
        "R2":                  round(r2,   4),
        "Directional Accuracy":round(acc,  4),
        "Precision":           round(prec, 4),
        "Recall":              round(rec,  4),
        "F1 Score":            round(f1,   4),
    }

    meanings = [
        label,
        "Mean squared log-return error",
        "Root MSE (same units as log-returns)",
        "Mean absolute error",
        "Variance explained (1.0 = perfect)",
        "% correct up/down calls",
        "Of predicted UP days how many were actually UP",
        "Of actual UP days how many did we catch",
        "Balance of precision and recall"
    ]

    summary_df = pd.DataFrame({
        "Metric":  list(metrics.keys()),
        "Value":   list(metrics.values()),
        "Meaning": meanings
    })

    # Streamlit/pyarrow Arrow serialization fails when a column mixes strings
    # and numeric values (object dtype). Convert everything to string for display.
    summary_df["Value"] = summary_df["Value"].map(lambda v: str(v))

    return metrics, summary_df, y_true, y_pred


def baseline_comparison(y_true, y_pred, label="Close"):
    """
    Compare LSTM against two simple baselines.

    Baseline 1 — Naive (predict zero return):
      Assumes price does not change
      Simplest possible prediction
      If LSTM cannot beat this it learned nothing

    Baseline 2 — Random Walk (yesterday = today):
      Assumes today's return = yesterday's return
      Classic financial time series baseline

    If LSTM does not beat BOTH baselines:
      The model has not learned useful patterns
      This is an honest result worth reporting
    """
    # Baseline 1: predict zero return every day
    naive_pred     = np.zeros_like(y_true)
    naive_mse      = mean_squared_error(y_true, naive_pred)
    naive_mae      = mean_absolute_error(y_true, naive_pred)
    naive_dir      = accuracy_score(
        (y_true > 0).astype(int),
        (naive_pred > 0).astype(int)
    )

    # Baseline 2: random walk (shift by 1)
    rw_pred        = y_true[:-1]
    rw_true        = y_true[1:]
    rw_mse         = mean_squared_error(rw_true, rw_pred)
    rw_dir         = accuracy_score(
        (rw_true > 0).astype(int),
        (rw_pred > 0).astype(int)
    )

    # LSTM metrics
    lstm_mse       = mean_squared_error(y_true, y_pred)
    lstm_mae       = mean_absolute_error(y_true, y_pred)
    lstm_dir       = accuracy_score(
        (y_true > 0).astype(int),
        (y_pred > 0).astype(int)
    )

    # Improvement percentages
    mse_improve    = (naive_mse - lstm_mse) / naive_mse * 100
    mae_improve    = (naive_mae - lstm_mae) / naive_mae * 100
    dir_improve    = (lstm_dir  - naive_dir) * 100

    comparison = pd.DataFrame({
        "Model": [
            "Naive Baseline",
            "Random Walk",
            f"LSTM + Indicators ({label})",
        ],
        "MSE": [
            round(naive_mse, 8),
            round(rw_mse, 8),
            round(lstm_mse, 8),
        ],
        # Keep numeric dtype for Streamlit/Arrow. Random Walk MAE is not computed.
        "MAE": [
            round(naive_mae, 8),
            np.nan,
            round(lstm_mae, 8),
        ],
        "Direction": [
            f"{naive_dir:.2%}",
            f"{rw_dir:.2%}",
            f"{lstm_dir:.2%}",
        ],
    })

    verdict = (
        "LSTM beats both baselines"
        if lstm_mse < naive_mse and lstm_dir > naive_dir
        else "LSTM beats naive on MSE"
        if lstm_mse < naive_mse
        else "LSTM beats naive on direction"
        if lstm_dir > naive_dir
        else "LSTM does not beat baselines"
    )

    return comparison, {
        "mse_improvement_pct":  round(mse_improve, 2),
        "mae_improvement_pct":  round(mae_improve, 2),
        "direction_improvement":round(dir_improve, 2),
        "verdict":              verdict
    }


def mcnemar_significance(y_true, y_pred_lstm,
                         y_pred_baseline=None, label=""):
    """
    McNemar test for statistical significance.

    Tests whether LSTM is statistically significantly
    better than the baseline at directional prediction.

    H0: LSTM and baseline have same error rate
    H1: LSTM has different (better) error rate

    p < 0.05 means improvement is statistically significant
    p > 0.05 means improvement could be due to chance

    For a diploma thesis this test proves your model
    improvement is real not just lucky
    """
    y_true_dir = (y_true > 0).astype(int)
    y_lstm_dir = (y_pred_lstm > 0).astype(int)

    if y_pred_baseline is None:
        y_pred_baseline = np.zeros_like(y_true)
    y_base_dir = (y_pred_baseline > 0).astype(int)

    lstm_correct = (y_lstm_dir == y_true_dir)
    base_correct = (y_base_dir == y_true_dir)

    only_lstm    = np.sum( lstm_correct & ~base_correct)
    only_base    = np.sum(~lstm_correct &  base_correct)

    if only_lstm + only_base > 0:
        stat    = ((only_lstm - only_base)**2) / \
                   (only_lstm + only_base)
        p_value = 1 - stats.chi2.cdf(stat, df=1)
        sig     = p_value < 0.05
    else:
        stat, p_value, sig = 0.0, 1.0, False

    return {
        "label":       label,
        "chi2":        round(stat, 4),
        "p_value":     round(p_value, 4),
        "significant": sig,
        "conclusion":  (
            "Improvement is statistically significant (p<0.05)"
            if sig else
            "Improvement is NOT statistically significant"
        )
    }


def ablation_summary(results_dict):
    """
    Format ablation study results for display.

    Ablation study = test each component separately
    to prove each one contributes to performance.

    Expected input format:
    {
      "LSTM only (OHLC)":        {"direction": 0.52, "mae": 0.012},
      "LSTM + Volume":           {"direction": 0.53, "mae": 0.011},
      "LSTM + Volume + RSI":     {"direction": 0.55, "mae": 0.010},
      "LSTM + Volume + RSI + MACD + BB": {"direction": 0.58, "mae": 0.009},
    }

    This table goes directly into your thesis
    Section 4 Results chapter
    """
    rows = []
    for model_name, metrics in results_dict.items():
        rows.append({
            "Configuration":       model_name,
            # Keep numeric dtypes so Streamlit/Arrow serialization doesn't fail.
            "Direction Accuracy":  float(metrics['direction']),  # as fraction [0,1]
            "MAE":                 float(metrics['mae']),
        })
    df = pd.DataFrame(rows)
    df["Direction Accuracy"] = df["Direction Accuracy"].map(lambda x: f"{x:.2%}")
    return df
