# ============================================================
# src/sentiment.py
# FinBERT + RoBERTa Sentiment Analysis & Fusion
# Per Academic Methodology Description
# ============================================================
import numpy as np
import torch
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

DEVICE = torch.device("cpu")

# ============================================================
# SCALING PARAMETERS (per description)
# ============================================================
ALPHA = 0.10       # Article scale: bounds individual article to [-0.10, +0.10]
BETA = 0.20        # Session scale: caps cumulative per session to [-0.20, +0.20]
FINBERT_WEIGHT = 0.60  # FinBERT weight (60%)
ROBERTA_WEIGHT = 0.40 # RoBERTa weight (40%)
TAU = 10           # Time decay tau = 10 days (exponential decay)


def load_nlp():
    """
    Load FinBERT and RoBERTa models.
    Call once and cache with @st.cache_resource in app.py.
    Returns: fin_tok, fin_mod, roberta_pipeline
    """
    fin_tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    fin_mod = AutoModelForSequenceClassification.from_pretrained(
        "yiyanghkust/finbert-tone"
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


def finbert_sentiment(text: str, fin_tok, fin_mod) -> Tuple[str, float]:
    """
    Run FinBERT on financial text.
    Returns: (label, confidence) where label in {Bearish, Neutral, Bullish}
    """
    inputs = fin_tok(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        out = fin_mod(**inputs)

    probs = torch.softmax(out.logits, dim=-1)[0]
    labels = ["Bearish", "Neutral", "Bullish"]
    idx = torch.argmax(probs).item()
    return labels[idx], probs[idx].item()


def roberta_sentiment(text: str, roberta) -> Tuple[str, float]:
    """
    Run RoBERTa on news text.
    Returns: (label, confidence)
    """
    result = roberta(text[:512])[0]
    return result["label"], result["score"]


def label_to_direction(label: str) -> float:
    """
    Convert label to directional score: +1 (Bullish), 0 (Neutral), -1 (Bearish)
    """
    label = label.upper()
    if "BULLISH" in label or "POSITIVE" in label or label == "POS":
        return +1.0
    if "BEARISH" in label or "NEGATIVE" in label or label == "NEG":
        return -1.0
    return 0.0


def fuse_sentiment(
    fin_label: str,
    fin_score: float,
    rob_label: str,
    rob_score: float,
) -> Tuple[float, bool]:
    """
    Weighted fusion of FinBERT (60%) + RoBERTa (40%) into a bias value.
    
    Args:
        fin_label: FinBERT label (Bullish/Bearish/Neutral)
        fin_score: FinBERT confidence
        rob_label: RoBERTa label
        rob_score: RoBERTa confidence
    
    Returns:
        Tuple of (bias in range [-ALPHA, +ALPHA], has_news (bool))
    
    Article scale parameter alpha=0.10 bounds individual article impact.
    """
    fin_dir = label_to_direction(fin_label)
    rob_dir = label_to_direction(rob_label)
    
    fin_signal = fin_dir * fin_score * FINBERT_WEIGHT
    rob_signal = rob_dir * rob_score * ROBERTA_WEIGHT
    
    fused = fin_signal + rob_signal
    
    bias = np.clip(fused * ALPHA, -ALPHA, +ALPHA)
    
    has_news = (fin_score > 0.5) or (rob_score > 0.5)
    
    return float(bias), has_news


def compute_time_decay_weight(pub_time: datetime, reference_time: datetime) -> float:
    """
    Exponential time-decay function with tau = 10 days.
    
    weight = exp(-age_days / tau)
    
    This modulates recency of each article's contribution.
    """
    if pub_time.tzinfo is None:
        pub_time = pub_time.replace(tzinfo=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    age_days = (reference_time - pub_time).total_seconds() / 86400
    return np.exp(-age_days / TAU)


def compute_session_sentiment(
    article_biases: List[float],
    has_news_flags: List[bool],
    publish_times: Optional[List[datetime]] = None,
    reference_time: Optional[datetime] = None,
) -> Tuple[float, bool]:
    """
    Compute cumulative sentiment for a trading session.
    
    Args:
        article_biases: List of per-article bias values
        has_news_flags: Binary indicator for news presence
        publish_times: Publication times (for time decay)
        reference_time: Reference datetime for decay calculation
    
    Returns:
        Tuple of (total_bias clipped to [-BETA, +BETA], has_news)
    
    - Binary indicator variable captures presence of news
    - Sessions without news = missing (not neutral)
    - Total scale parameter beta=0.20 caps cumulative contribution
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    
    has_news = any(has_news_flags)
    
    if not article_biases:
        return 0.0, False
    
    biases = np.array(article_biases)
    
    if publish_times and len(publish_times) == len(biases):
        weights = np.array([
            compute_time_decay_weight(pt, reference_time)
            for pt in publish_times
        ])
        weights = weights / (weights.sum() + 1e-9)
        total = float(np.dot(weights, biases))
    else:
        total = float(np.mean(biases))
    
    total_clipped = np.clip(total, -BETA, BETA)
    
    return total_clipped, has_news


def analyze_news(
    headlines: List[str],
    publish_times: Optional[List[datetime]] = None,
    fin_tok=None,
    fin_mod=None,
    roberta=None,
) -> Dict:
    """
    Main entry point: analyze a list of headlines.
    
    Args:
        headlines: List of news headline strings
        publish_times: List of publication datetimes (optional)
        fin_tok, fin_mod, roberta: Pre-loaded models (optional)
    
    Returns:
        Dict with:
            - total_bias: Cumulative sentiment in [-BETA, +BETA]
            - signal: "Bullish"|"Bearish"|"Neutral"
            - has_news: bool indicating news was present
            - article_scores: List of per-article results
    """
    if fin_tok is None or fin_mod is None or roberta is None:
        fin_tok, fin_mod, roberta = load_nlp()
    
    if publish_times is None:
        publish_times = [datetime.now(timezone.utc)] * len(headlines)
    
    article_biases = []
    has_news_flags = []
    article_scores = []
    
    for headline in headlines:
        fin_label, fin_score = finbert_sentiment(headline, fin_tok, fin_mod)
        rob_label, rob_score = roberta_sentiment(headline, roberta)
        
        bias, has_news = fuse_sentiment(fin_label, fin_score, rob_label, rob_score)
        
        article_biases.append(bias)
        has_news_flags.append(has_news)
        
        article_scores.append({
            "text": headline[:80] + "...",
            "finbert": fin_label,
            "roberta": rob_label,
            "bias": bias,
            "has_news": has_news,
        })
    
    total_bias, has_news = compute_session_sentiment(
        article_biases,
        has_news_flags,
        publish_times,
    )
    
    if total_bias > 0.02:
        signal = "Bullish"
    elif total_bias < -0.02:
        signal = "Bearish"
    else:
        signal = "Neutral"
    
    return {
        "total_bias": total_bias,
        "signal": signal,
        "has_news": has_news,
        "article_scores": article_scores,
        "n_articles": len(headlines),
    }


# ============================================================
# Backwards compatibility aliases
# ============================================================
def compute_total_bias(article_biases, clip_range=0.2):
    """
    Legacy compatibility: compute total bias with clip range.
    clip_range corresponds to BETA parameter.
    """
    return float(np.clip(sum(article_biases), -clip_range, clip_range))