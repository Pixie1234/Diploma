# LSTM Stock Prediction - Evaluation Results

## 1. Real News Results (Finlight API)

### Stock 1: AAPL (Apple)

| Model | RMSE | MAE | Directional Accuracy |
|-------|------|----|---------------------|
| LSTM | 5.35 | 4.09 | 51.7% |
| LSTM + Technical Indicators | 11.89 | 10.47 | 48.3% |
| **Proposed Model** (with news sentiment) | -- | -- | **58.6%** |

Sentiment: Bearish (-0.0315)

### Stock 2: JPM (JPMorgan)

| Model | RMSE | MAE | Directional Accuracy |
|-------|------|----|---------------------|
| LSTM | 18.41 | 17.25 | 55.2% |
| LSTM + Technical Indicators | 28.64 | 24.45 | 44.8% |
| **Proposed Model** (with news sentiment) | -- | -- | **55.2%** |

Sentiment: Neutral (+0.0041)

---

## 2. Proof of Concept Results (Synthetic Sentiment)

Using price momentum as sentiment proxy:

| Stock | LSTM+Tech | Proposed (Synthetic) |
|-------|-----------|---------------------|
| AAPL | 58.6% | 65.5% |
| MSFT | 51.7% | 48.3% |
| NVDA | 51.7% | 58.6% |
| XOM | 62.1% | 62.1% |
| JPM | 48.3% | 58.6% |
| **Average** | **54.5%** | **58.6%** |

---

## Summary

### Real News (Finlight)

| Stock | LSTM | LSTM+Tech | Proposed |
|-------|------|-----------|----------|
| AAPL | 51.7% | 48.3% | **58.6%** |
| JPM | 55.2% | 44.8% | **55.2%** |

### Proof of Concept (5 Stocks)

| Metric | LSTM+Tech | Proposed |
|--------|-----------|----------|
| Average Dir Acc | 54.5% | **58.6%** |

---

## Methodology

- **Data Split:** 70% Train / 15% Validation / 15% Test (temporal)
- **News Source:** Finlight API (real financial news)
- **Sentiment Model:** FinBERT (60%) + RoBERTa (40%) fusion
- **Test Period:** 30 days

## Key Findings

1. **With real news (AAPL):** +6.9% improvement
2. **With synthetic sentiment:** +4.1% average improvement
3. Proposed model outperforms baseline in majority of cases

## Resource Constraints

Financial news APIs have limited historical data. Free APIs provide only recent news. Paid APIs (Bloomberg, Refinitiv) offer historical data but cost $15K-25K+/year - not accessible for academic research. Therefore, a proof of concept was developed using price momentum as sentiment proxy, demonstrating +4.1% average improvement to validate the methodology.