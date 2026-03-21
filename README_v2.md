# Candlestick Pattern Classification using CNN + Backtesting

## 1. Overview
This project applies a Convolutional Neural Network (CNN) to classify candlestick patterns from stock chart images. It integrates a complete pipeline from data acquisition and image generation to model training and backtesting, evaluating the predictive power of technical patterns in a quantitative trading context.

## 2. Problem Statement
The goal is to automate the detection of specific candlestick patterns (Marubozu and Shooting Star) directly from OHLC (Open, High, Low, Close) data converted into images. Furthermore, the project assesses whether these classical patterns, when identified by a CNN, provide a profitable signal for short-term trading.

## 3. Dataset
*   **Source:** Historical stock data downloaded via Yahoo Finance (`yfinance`).
*   **Preprocessing:** OHLC data converted to candlestick chart images.
*   **Generation:** Uses a sliding window of 20 candles per image.
*   **Labeling:** Rule-based logic used to ground-truth samples.
*   **Classes:**
    *   `Marubozu` (Strong trend)
    *   `Shooting Star` (Bearish reversal)

## 4. Methodology
1.  **Data Ingestion:** Fetch daily stock data for a diverse set of tickers.
2.  **Image Synthesis:** Generate 224x224 candlestick charts without axes/volume to focus solely on price action.
3.  **Model Architecture:** Train a custom CNN from scratch (no pretrained weights) to classify the images.
4.  **Evaluation:** Test on a held-out dataset and validate performance using accuracy metrics and confusion matrices.

## 5. Model Performance
*   **Test Accuracy:** ~82%
*   **Confusion Matrix:** See `confusion_matrix.png` for detailed class-wise performance.

## 6. Backtesting Strategy
*   **Entry Signal:** Buy at the next day's Open price if a pattern is detected with high confidence.
*   **Exit Signal:** Sell at the Close price after 3 trading days.
*   **Constraint:** Sequential trades only (no overlapping positions).
*   **Transaction Costs:** 0.1% per trade applied to simulate real-world friction.

## 7. Backtesting Results
*   **CAGR:** ~22%
*   **Total Return:** ~22%
*   **Win Rate:** ~51%
*   **Max Drawdown:** ~28%
*   **Total Trades:** 57

## 8. Key Insights
*   **Pattern Efficacy:** The *Shooting Star* pattern demonstrated positive expected returns in the test period.
*   **False Signals:** The *Marubozu* pattern, while detected accurately, was not profitable in this specific strategy.
*   **General Finding:** Classical technical patterns do not guarantee predictive value; their utility varies significantly by market regime and timeframe.

## 9. Limitations
*   **Dataset Size:** Limited number of samples compared to large-scale deep learning datasets.
*   **Labeling Bias:** Ground truth relies on fixed heuristic rules, which the model attempts to approximate.
*   **Scope:** Single timeframe (Daily) and limited asset universe.
*   **Risk Management:** Strategy lacks position sizing or dynamic stop-loss mechanisms.

## 10. Future Work
*   Expand dataset to include hundreds of tickers and multiple timeframes.
*   Incorporate additional patterns (e.g., Engulfing, Morning Star).
*   Implement portfolio allocation and risk management layers.
*   Experiment with advanced architectures (ResNet, LSTM-CNN).

## 11. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Generate Dataset
Download data and create images:
```bash
python src/generate_dataset.py
```

### 2. Train Model
Train the CNN and save the model:
```bash
python src/train_cnn.py
```

### 3. Run Backtest
Evaluate the strategy:
```bash
python src/backtest.py
```
