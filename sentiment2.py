# ============================================================
# sentiment2.py  —  FinBERT + RoBERTa Sentiment Analysis
# ============================================================
#
# KEY IMPROVEMENTS:
#  1. CONFIDENCE-WEIGHTED fusion
#  2. AGREEMENT BONUS (1.25×) / DISAGREEMENT DAMP (0.5×)
#  3. RECENCY DECAY at article level
#  4. VOLATILITY-ADAPTIVE SCALING
#  5. MOMENTUM CONSISTENCY CHECK
#  6. ASYMMETRIC CLIPPING
# ============================================================

import numpy as np
import torch
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

DEVICE = torch.device("cpu")

# Configuration
FINBERT_WEIGHT   = 0.60
ROBERTA_WEIGHT   = 0.40
AGREEMENT_BOOST  = 1.25
DISAGREEMENT_DAMP = 0.50
RECENCY_HALFLIFE = 1.5
MAX_BIAS_POS    = +0.12
MAX_BIAS_NEG    = -0.08
CONFLICT_PENALTY = 0.00

# Extra safety for forecast impact.
# The sentiment signal is mapped onto log-return space each day.
# Without a tight cap, even moderate bias can compound into large
# drifts over multi-day forecasts.
IMPACT_CLIP_POS = +0.01
IMPACT_CLIP_NEG = -0.01
IMPACT_CLIP_POS_CLOSE = +0.005
IMPACT_CLIP_NEG_CLOSE = -0.005


# ──────────────────────────────────────────────
# 1. Model loading
# ──────────────────────────────────────────────

def load_nlp():
    """
    Load FinBERT (ProsusAI/finbert) and RoBERTa.
    Returns: fin_tok, fin_mod, roberta_pipeline
    
    ProsusAI/finbert gives more accurate financial sentiment.
    """
    fin_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fin_mod = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert"
    ).to(DEVICE)
    fin_mod.eval()

    roberta = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,
        truncation=True,
        max_length=512,
    )
    return fin_tok, fin_mod, roberta


# ──────────────────────────────────────────────
# 2. Individual model inference
# ──────────────────────────────────────────────

def finbert_sentiment(text: str, fin_tok, fin_mod) -> Tuple[str, float]:
    """
    Run ProsusAI/FinBERT on financial text.
    Returns: (label, confidence)
    
    Model labels: idx 0=positive, 1=negative, 2=neutral
    """
    inputs = fin_tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        out = fin_mod(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0]
    
    # ProsusAI/finbert: idx 0=positive, 1=negative, 2=neutral
    pos, neg, neu = probs[0].item(), probs[1].item(), probs[2].item()
    
    if pos >= neg and pos >= neu:
        return "Bullish", pos
    elif neg >= neu:
        return "Bearish", neg
    else:
        return "Neutral", neu


def roberta_sentiment(text: str, roberta) -> Tuple[str, float]:
    """Run RoBERTa on text. Returns: (label, confidence)"""
    result = roberta(text[:512])[0]
    return result["label"], float(result["score"])


# ──────────────────────────────────────────────
# 3. Label normalisation
# ──────────────────────────────────────────────

def _label_to_direction(label: str) -> int:
    """Convert any label → {+1, 0, -1}"""
    u = label.upper()
    if any(k in u for k in ["BULLISH", "POSITIVE", "POS", "LABEL_2"]):
        return +1
    if any(k in u for k in ["BEARISH", "NEGATIVE", "NEG", "LABEL_0"]):
        return -1
    return 0


# ──────────────────────────────────────────────
# 4. Per-article fusion
# ──────────────────────────────────────────────

def fuse_sentiment(
    fin_label: str,
    fin_score: float,
    rob_label: str,
    rob_score: float,
) -> float:
    """
    Confidence-weighted fusion with agreement bonus/disagreement damp.
    Returns raw bias in [-0.10, +0.10].
    """
    fin_dir = _label_to_direction(fin_label)
    rob_dir = _label_to_direction(rob_label)

    fin_signal = fin_dir * fin_score * FINBERT_WEIGHT
    rob_signal = rob_dir * rob_score * ROBERTA_WEIGHT

    fused = fin_signal + rob_signal

    if fin_dir != 0 and rob_dir != 0:
        if fin_dir == rob_dir:
            fused *= AGREEMENT_BOOST
        else:
            fused *= DISAGREEMENT_DAMP

    # Scale the raw fused signal down so that adding sentiment
    # shifts forecasts modestly instead of dominating daily returns.
    fused_scaled = fused * 0.01
    fused_clipped = float(np.clip(fused_scaled, MAX_BIAS_NEG, MAX_BIAS_POS))
    return fused_clipped


# ──────────────────────────────────────────────
# 5. Recency-weighted aggregation
# ──────────────────────────────────────────────

def compute_total_bias(
    article_biases: List[float],
    publish_times: Optional[List[datetime]] = None,
    clip_range: float = 0.20,
    reference_time: Optional[datetime] = None,
) -> float:
    """
    Aggregate per-article biases with recency weighting.
    If publish_times provided, weight by age.
    """
    if not article_biases:
        return 0.0

    biases = np.array(article_biases, dtype=float)

    if publish_times and len(publish_times) == len(biases):
        # Some callers may pass None for missing publish timestamps.
        # Skip those pairs for recency weighting (no timestamp → no recency info).
        known_mask = [t is not None for t in publish_times]
        if any(known_mask):
            biases_known = biases[np.array(known_mask, dtype=bool)]
            publish_times_known = [t for t, ok in zip(publish_times, known_mask) if ok]

            now = reference_time if reference_time is not None else datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            ages_days = np.array([
                max(0.0, (now - _ensure_utc(t)).total_seconds() / 86400)
                for t in publish_times_known
            ])
            weights = np.exp(-ages_days / RECENCY_HALFLIFE)
            weights = weights / (weights.sum() + 1e-9)
            total = float(np.dot(weights, biases_known))
        else:
            total = float(np.mean(biases))
    else:
        total = float(np.mean(biases))

    return float(np.clip(total, -clip_range, clip_range))


def _ensure_utc(dt: datetime) -> datetime:
    """Make datetime tz-aware (UTC)"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ──────────────────────────────────────────────
# 6. Volatility-adaptive + momentum-consistent fusion
# ──────────────────────────────────────────────

def apply_sentiment_fusion(
    total_bias: float,
    open_returns: List[float],
    close_returns: List[float],
    last_open: float,
    last_close: float,
    days_to_predict: int,
    recent_returns: Optional[np.ndarray] = None,
    forecast_decay: float = 10.0,
) -> Dict[str, List[float]]:
    """
    Apply sentiment bias to LSTM forecasts.

    Returns dict:
        "fused_open": adjusted Open prices
        "fused_close": adjusted Close prices
        "impact_curve": sentiment impact per day
    """
    # Volatility-adaptive scale
    if recent_returns is not None and len(recent_returns) >= 5:
        realised_vol = float(np.std(recent_returns[-20:]))
        # Cap volatility scaling to avoid pushing sentiment impact to the
        # clipping bounds too aggressively (which can over-dampen forecasts).
        vol_scale = np.clip(realised_vol / 0.01, 0.5, 1.0)
    else:
        vol_scale = 1.0

    scaled_bias = total_bias * vol_scale
    decay = np.exp(-np.arange(days_to_predict) / forecast_decay)

    fused_open = []
    fused_close = []
    impact_curve = []

    p_open = last_open
    p_close = last_close

    for i in range(days_to_predict):
        o_ret = open_returns[i]
        c_ret = close_returns[i]

        # Momentum consistency check
        lstm_direction = np.sign(c_ret) if abs(c_ret) > 1e-6 else 0
        sentiment_direction = np.sign(scaled_bias)

        if lstm_direction != 0 and sentiment_direction != 0:
            if lstm_direction == sentiment_direction:
                consistency_factor = 1.0
            else:
                consistency_factor = CONFLICT_PENALTY
        else:
            consistency_factor = 1.0

        impact_base = scaled_bias * decay[i] * consistency_factor
        impact_open = float(np.clip(impact_base, IMPACT_CLIP_NEG, IMPACT_CLIP_POS))
        impact_close = float(
            np.clip(impact_base, IMPACT_CLIP_NEG_CLOSE, IMPACT_CLIP_POS_CLOSE)
        )

        p_open *= np.exp(o_ret + impact_open)
        p_close *= np.exp(c_ret + impact_close)

        fused_open.append(round(p_open, 4))
        fused_close.append(round(p_close, 4))
        impact_curve.append(round(impact_close, 6))

    return {
        "fused_open": fused_open,
        "fused_close": fused_close,
        "impact_curve": impact_curve,
    }


# ──────────────────────────────────────────────
# 7. Batch scoring
# ──────────────────────────────────────────────

def score_articles_batch(
    texts: List[str],
    fin_tok,
    fin_mod,
    roberta,
    pub_times: Optional[List[datetime]] = None,
) -> Dict:
    """Score multiple articles and return results."""
    biases = []
    scores_out = []

    for text in texts:
        fin_l, fin_s = finbert_sentiment(text, fin_tok, fin_mod)
        rob_l, rob_s = roberta_sentiment(text, roberta)
        bias = fuse_sentiment(fin_l, fin_s, rob_l, rob_s)
        biases.append(bias)
        scores_out.append({
            "text_preview": text[:80] + "…",
            "finbert_label": fin_l,
            "finbert_conf": round(fin_s, 4),
            "roberta_label": rob_l,
            "roberta_conf": round(rob_s, 4),
            "fused_bias": round(bias, 6),
        })

    total = compute_total_bias(biases, publish_times=pub_times)
    signal = (
        "Bullish" if total > 0.02 else
        "Bearish" if total < -0.02 else
        "Neutral"
    )
    return {
        "total_bias": total,
        "article_scores": scores_out,
        "signal": signal,
    }
