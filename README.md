# Diploma Thesis — Stock Price Prediction with Sentiment

This repository contains the code for a diploma thesis on **Open/Close stock price prediction using a multimodal model that combines price data and real-news sentiment**.

## Evaluation Stocks

Model evaluation and backtesting are performed on **three stocks**:

- **MMM**
- **NVDA**
- **TSLA (Tesla)**

## What’s Included

- Streamlit UI (`app2.py`) to run inference, view forecasts, and inspect test-set quality.
- Data pipeline and model code for training and inference.
- Sentiment integration logic (FinBERT + RoBERTa) and supporting utilities.
- Multimodal input-feature evaluation scripts for sentiment-aware forecasting.

## How to Run

- Start the Streamlit app from the project directory.
- Use the UI to select the ticker and run the prediction workflow.
