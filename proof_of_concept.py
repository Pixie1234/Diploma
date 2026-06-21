#!/usr/bin/env python3
"""
Proof of Concept: Synthetic Sentiment from Price Momentum

Instead of using real news sentiment, we use price momentum as proxy:
- If price rose yesterday → bullish sentiment
- If price fell yesterday → bearish sentiment
- Scale factor = 20, clipped to [-0.20, +0.20]
"""

import numpy as np


def synthetic_sentiment_from_price(close_prices, current_idx):
    """
    Generate synthetic sentiment from previous day price movement.
    
    Args:
        close_prices: Array of close prices
        current_idx: Index for prediction (t)
    
    Returns:
        float: Sentiment in range [-0.20, +0.20]
    """
    BETA = 0.20
    SCALE_FACTOR = 20.0
    
    if current_idx < 2:
        return 0.0
    
    # Previous day close
    prev_close = close_prices[current_idx - 1]
    # Day before close
    prev_prev_close = close_prices[current_idx - 2]
    
    # Log return (percentage change)
    log_return = np.log(prev_close / (prev_prev_close + 1e-8))
    
    # Scale and clip
    sentiment = np.clip(log_return * SCALE_FACTOR, -BETA, BETA)
    
    return float(sentiment)


# Example usage
if __name__ == "__main__":
    # Simulated close prices (e.g., 5 days)
    close_prices = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
    
    print("Proof of Concept: Synthetic Sentiment")
    print("=" * 40)
    
    for i in range(2, len(close_prices)):
        sentiment = synthetic_sentiment_from_price(close_prices, i)
        
        # What happened
        prev_close = close_prices[i - 1]
        prev_prev_close = close_prices[i - 2]
        pct_change = ((prev_close - prev_prev_close) / prev_prev_close) * 100
        
        # Label
        if sentiment > 0.05:
            label = "BULLISH"
        elif sentiment < -0.05:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        
        print(f"Day {i}: {pct_change:+.2f}% → Sentiment: {sentiment:+.3f} ({label})")