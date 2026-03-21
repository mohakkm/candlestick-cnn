"""
backtest.py
===========
This script backtests the trained CNN model on new stock data.

Trading Strategy:
- When a pattern is detected, buy at next day's open
- Sell at close after 3 trading days
- Calculate total return and win rate
"""

import os
import io
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
from PIL import Image
from torchvision import transforms

# ============================================================
# CONFIGURATION
# ============================================================

# Model path
MODEL_PATH = 'candlestick_cnn_model.pth'

# Stocks to backtest on (different from training data)
BACKTEST_TICKERS = ['BA', 'CAT', 'GE', 'IBM', 'MCD']

# Window size (must match training)
WINDOW_SIZE = 20

# Image size (must match training)
IMAGE_SIZE = 224

# Trading parameters
HOLD_DAYS = 3          # How many days to hold position
INITIAL_CAPITAL = 10000  # Starting capital
TRANSACTION_COST = 0.001  # 0.1% per trade (entry + exit combined)

# Number of classes
NUM_CLASSES = 2
CLASS_NAMES = ['marubozu', 'shooting_star']  # Alphabetical order from ImageFolder

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# CNN MODEL (same as train_cnn.py)
# ============================================================

class SimpleCNN(nn.Module):
    """
    Improved CNN architecture with better capacity to handle all patterns.
    
    Key improvements:
    - More filters in each layer for better feature extraction
    - Batch normalization to stabilize training
    - More dropout to reduce overfitting on Doji
    - Larger fully connected layer
    """
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # ===== CONVOLUTIONAL LAYERS =====
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # ===== POOLING AND ACTIVATION =====
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # ===== FULLY CONNECTED LAYERS =====
        # After 3 pooling: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # ===== DROPOUT (increased to prevent overfitting) =====
        self.dropout1 = nn.Dropout(0.6)  # After conv layers
        self.dropout2 = nn.Dropout(0.5)  # After first FC layer
        self.dropout3 = nn.Dropout(0.3)  # After second FC layer
    
    def forward(self, x):
        """Forward pass with batch norm and increased regularization."""
        # Layer 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Layer 2: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        # Layer 3: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout3(x)
        
        x = self.fc3(x)
        
        return x


# ============================================================
# DATA FUNCTIONS
# ============================================================

def download_backtest_data(ticker, start_date, end_date):
    """
    Download stock data for backtesting.
    """
    print(f"  Downloading {ticker}...")
    
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if df.empty:
            print(f"  Warning: No data for {ticker}")
            return None
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Open', 'High', 'Low', 'Close']]
        df = df.dropna()
        
        print(f"  Got {len(df)} days of data")
        return df
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def create_candlestick_image_in_memory(df_window):
    """
    Create a candlestick image in memory (not saved to disk).
    
    Returns: PIL Image object
    """
    # Create custom style
    custom_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        gridstyle='',
    )
    
    fig_size = (IMAGE_SIZE / 100, IMAGE_SIZE / 100)
    
    # Create figure and save to buffer
    buf = io.BytesIO()
    
    mpf.plot(
        df_window,
        type='candle',
        style=custom_style,
        axisoff=True,
        volume=False,
        savefig=dict(
            fname=buf,
            dpi=100,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='black',
            format='png'
        ),
        figsize=fig_size
    )
    
    # Read image from buffer
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    
    return image


def preprocess_image(image):
    """
    Preprocess image for CNN input.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Transform and add batch dimension
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)
    
    return tensor


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_pattern(model, df_window):
    """
    Use the CNN to predict the pattern in a candlestick window.
    
    Returns: (predicted_class_name, confidence)
    """
    # Create image
    image = create_candlestick_image_in_memory(df_window)
    
    # Preprocess
    tensor = preprocess_image(image)
    tensor = tensor.to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_name = CLASS_NAMES[predicted.item()]
    conf = confidence.item()
    
    return class_name, conf


# ============================================================
# BACKTESTING LOGIC
# ============================================================

class Trade:
    """
    Stores all information for a single completed trade.

    Timeline (index-based, no lookahead):
      signal_date : Close of candle at index i-1  (last candle in the window)
      entry_date  : Open  of candle at index i     (next day, first tradeable bar)
      exit_date   : Close of candle at index i + HOLD_DAYS
    """
    def __init__(self, signal_date, entry_date, entry_price, pattern, confidence):
        self.signal_date        = signal_date
        self.entry_date         = entry_date
        self.entry_price        = entry_price
        self.pattern            = pattern
        self.confidence         = confidence
        self.ticker             = None   # will be set by backtest function
        self.exit_date          = None
        self.exit_price         = None
        self.gross_return_pct   = None   # raw, before costs
        self.net_return_pct     = None   # after 0.1% cost
        self.return_after_costs = None   # decimal, for compounding
        self.capital_after      = None   # filled in by calculate_results

    def close(self, exit_date, exit_price, transaction_cost):
        """Record exit and compute gross / net returns."""
        self.exit_date = exit_date
        self.exit_price = exit_price

        gross = (exit_price - self.entry_price) / self.entry_price   # decimal
        self.gross_return_pct   = gross * 100
        self.return_after_costs = gross - transaction_cost            # decimal, net
        self.net_return_pct     = self.return_after_costs * 100


def backtest_all_stocks(model, stock_data, min_confidence=0.5):
    """
    Backtest across all stocks with ONLY ONE active trade at a time (globally).
    
    Strategy:
    - Scan all stocks day by day
    - When a signal is found, enter trade immediately
    - Hold for HOLD_DAYS, then exit
    - Resume scanning from exit_date + 1
    """
    all_trades = []
    
    # Combine all stocks into one timeline with dates
    combined_timeline = []
    for ticker, df in stock_data.items():
        if len(df) < WINDOW_SIZE + HOLD_DAYS:
            continue
        dates = df.index.tolist()
        for i in range(WINDOW_SIZE, len(df) - HOLD_DAYS):
            combined_timeline.append({
                'date': dates[i],
                'ticker': ticker,
                'index': i
            })
    
    # Sort by date
    combined_timeline.sort(key=lambda x: x['date'])
    
    print(f"\n  Scanning {len(stock_data)} stocks with ONE position at a time...")
    
    # Process timeline with global position tracking
    position_active = False
    current_exit_date = None
    idx = 0
    
    while idx < len(combined_timeline):
        opportunity = combined_timeline[idx]
        ticker = opportunity['ticker']
        i = opportunity['index']
        df = stock_data[ticker]
        dates = df.index.tolist()
        
        # Skip if we're in an active position
        if position_active and opportunity['date'] <= current_exit_date:
            idx += 1
            continue
        
        # Position is now free
        position_active = False
        
        # Build signal window
        df_window = df.iloc[i - WINDOW_SIZE : i].copy()
        
        try:
            pattern, confidence = predict_pattern(model, df_window)
        except Exception:
            idx += 1
            continue
        
        if confidence < min_confidence:
            idx += 1
            continue
        
        # Valid signal found - enter trade
        signal_candle_idx = i - 1
        entry_idx = i
        exit_idx = i + HOLD_DAYS
        
        if exit_idx >= len(df):
            idx += 1
            continue
        
        signal_date = dates[signal_candle_idx]
        entry_date = dates[entry_idx]
        exit_date = dates[exit_idx]
        entry_price = float(df.iloc[entry_idx]['Open'])
        exit_price = float(df.iloc[exit_idx]['Close'])
        
        # Create and close trade
        trade = Trade(signal_date, entry_date, entry_price, pattern, confidence)
        trade.close(exit_date, exit_price, TRANSACTION_COST)
        trade.ticker = ticker  # Add ticker info
        all_trades.append(trade)
        
        # Mark position as active and set exit date
        position_active = True
        current_exit_date = exit_date
        
        # Continue scanning from next day
        idx += 1
    
    return all_trades


def calculate_results(all_trades, initial_capital, start_date, end_date):
    """
    Compound capital trade-by-trade (after transaction costs).
    Computes CAGR, max drawdown, win rate.
    Formula per trade:
        capital = capital * (1 + net_return_decimal)
        net_return_decimal = gross_return - TRANSACTION_COST
    """
    empty = {
        'total_trades':   0,
        'winning_trades': 0,
        'losing_trades':  0,
        'win_rate':       0.0,
        'initial_capital': initial_capital,
        'final_capital':   initial_capital,
        'total_return':    0.0,
        'cagr':            0.0,
        'max_drawdown':    0.0,
        'avg_net_return':  0.0,
        'start_date':      start_date,
        'end_date':        end_date,
    }
    if len(all_trades) == 0:
        return empty

    # ── Compound capital and build equity curve ───────────────────────────
    capital       = float(initial_capital)
    capital_curve = [capital]

    for trade in all_trades:
        # capital = capital * (1 + net_return)  — the ONLY compounding formula used
        capital = capital * (1.0 + trade.return_after_costs)
        trade.capital_after = round(capital, 2)
        capital_curve.append(capital)

    final_capital = capital

    # ── Total compounded return ───────────────────────────────────────────
    total_return = (final_capital - initial_capital) / initial_capital * 100.0

    # ── CAGR ─────────────────────────────────────────────────────────────
    days_elapsed   = (end_date - start_date).days
    years_elapsed  = days_elapsed / 365.25
    cagr = (pow(final_capital / initial_capital, 1.0 / years_elapsed) - 1.0) * 100.0 \
           if years_elapsed > 0 and final_capital > 0 else 0.0

    # ── Maximum drawdown ─────────────────────────────────────────────────
    curve       = np.array(capital_curve)
    running_max = np.maximum.accumulate(curve)
    drawdowns   = (curve - running_max) / running_max * 100.0
    max_drawdown = float(np.min(drawdowns))

    # ── Win / loss counts (on gross return — cost is fixed, not a signal) ─
    winning = [t for t in all_trades if t.gross_return_pct > 0]
    losing  = [t for t in all_trades if t.gross_return_pct <= 0]

    return {
        'total_trades':   len(all_trades),
        'winning_trades': len(winning),
        'losing_trades':  len(losing),
        'win_rate':       len(winning) / len(all_trades) * 100.0,
        'initial_capital': initial_capital,
        'final_capital':   round(final_capital, 2),
        'total_return':    round(total_return, 4),
        'cagr':            round(cagr, 4),
        'max_drawdown':    round(max_drawdown, 4),
        'avg_net_return':  round(float(np.mean([t.net_return_pct for t in all_trades])), 4),
        'start_date':      start_date,
        'end_date':        end_date,
    }


def print_first_n_trades(all_trades, n=5):
    """
    Print the first n trades in detail for manual verification.
    Shows every date and price used so lookahead can be audited by eye.
    """
    print("\n" + "=" * 100)
    print(f"FIRST {n} TRADES — VERIFICATION (check: entry_date > signal_date, exit = entry + {HOLD_DAYS} bars)")
    print("=" * 100)
    hdr = (f"{'#':<4} {'Ticker':<6} {'Signal Date':<14} {'Entry Date':<14} {'Exit Date':<14}"
           f" {'Pattern':<14} {'Entry $':>9} {'Exit $':>9}"
           f" {'Gross%':>8} {'Net%':>8} {'Capital After':>14}")
    print(hdr)
    print("-" * 100)
    for idx, t in enumerate(all_trades[:n], 1):
        print(f"{idx:<4} "
              f"{t.ticker:<6} "
              f"{str(t.signal_date.date()):<14} "
              f"{str(t.entry_date.date()):<14} "
              f"{str(t.exit_date.date()):<14} "
              f"{t.pattern:<14} "
              f"${t.entry_price:>8.2f} "
              f"${t.exit_price:>8.2f} "
              f"{t.gross_return_pct:>+7.2f}% "
              f"{t.net_return_pct:>+7.2f}% "
              f"${t.capital_after:>13,.2f}")


def print_trade_summary(all_trades):
    """
    Print all trades in compact form.
    """
    print("\n" + "=" * 90)
    print("FULL TRADE LIST")
    print("=" * 90)
    print(f"{'Ticker':<6} {'Signal':<12} {'Entry':<12} {'Exit':<12} {'Pattern':<14}"
          f" {'Entry $':>9} {'Exit $':>9} {'Gross%':>8} {'Net%':>8}")
    print("-" * 90)
    for t in all_trades:
        print(f"{t.ticker:<6} "
              f"{str(t.signal_date.date()):<12} "
              f"{str(t.entry_date.date()):<12} "
              f"{str(t.exit_date.date()):<12} "
              f"{t.pattern:<14} "
              f"${t.entry_price:>8.2f} "
              f"${t.exit_price:>8.2f} "
              f"{t.gross_return_pct:>+7.2f}% "
              f"{t.net_return_pct:>+7.2f}%")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """
    Main function to run the backtest.
    """
    print("=" * 60)
    print("CANDLESTICK PATTERN CNN BACKTEST")
    print("=" * 60)
    
    # Step 1: Load the trained model
    print("\n" + "=" * 60)
    print("STEP 1: LOADING MODEL")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train_cnn.py first to train the model.")
        return
    
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Step 2: Download backtest data
    print("\n" + "=" * 60)
    print("STEP 2: DOWNLOADING DATA")
    print("=" * 60)
    
    # Use more recent data for backtesting
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Last 1 year
    
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    stock_data = {}
    for ticker in BACKTEST_TICKERS:
        df = download_backtest_data(ticker, start_date, end_date)
        if df is not None:
            stock_data[ticker] = df
    
    # Step 3: Run backtest
    print("\n" + "=" * 60)
    print("STEP 3: RUNNING BACKTEST")
    print("=" * 60)
    print(f"Hold period: {HOLD_DAYS} days")
    print(f"Min confidence: 0.5")
    print(f"Constraint: ONE position at a time across all stocks")
    
    all_trades = backtest_all_stocks(model, stock_data)

    # Step 4: Calculate and print results
    print("\n" + "=" * 60)
    print("STEP 4: RESULTS")
    print("=" * 60)

    if len(all_trades) == 0:
        print("\nNo trades were executed!")
        print("  - No patterns were detected with confidence >= 0.5")
        print("  - Try lowering the minimum confidence threshold")
        return

    # ── Sanity checks ────────────────────────────────────────────────────
    trading_days_in_period = (end_date - start_date).days
    holding_period_per_trade = HOLD_DAYS + 1  # Entry day + hold days
    max_possible_trades = trading_days_in_period // holding_period_per_trade
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    print(f"  Total calendar days     : {trading_days_in_period}")
    print(f"  Hold period per trade   : {holding_period_per_trade} days (entry + {HOLD_DAYS} hold)")
    print(f"  Max possible trades     : {max_possible_trades} (one at a time globally)")
    print(f"  Actual trades executed  : {len(all_trades)}")
    print(f"  Transaction cost        : {TRANSACTION_COST * 100}% per trade (both sides combined)")

    # ── Calculate stats (compounding happens here, capital_after set per trade) ─
    results = calculate_results(all_trades, INITIAL_CAPITAL, start_date, end_date)

    # ── First 5 trades verification ──────────────────────────────────────
    print_first_n_trades(all_trades, n=5)

    # ── Full trade list ───────────────────────────────────────────────────
    print_trade_summary(all_trades)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY (COMPOUNDED RETURNS)")
    print("=" * 60)

    print("\nBacktest Period:")
    print(f"  Start Date : {results['start_date'].date()}")
    print(f"  End Date   : {results['end_date'].date()}")
    print(f"  Duration   : {(results['end_date'] - results['start_date']).days} calendar days")

    print("\nCapital:")
    print(f"  Initial Capital : ${results['initial_capital']:>12,.2f}")
    print(f"  Final Capital   : ${results['final_capital']:>12,.2f}")
    print(f"  Profit / Loss   : ${results['final_capital'] - results['initial_capital']:>+12,.2f}")

    print("\nReturns  [formula: capital *= (1 + gross_return - 0.001) per trade]:")
    print(f"  Total Compounded Return : {results['total_return']:>+8.2f}%")
    print(f"  CAGR (Annualized)       : {results['cagr']:>+8.2f}%")
    print(f"  Avg Net Return/Trade    : {results['avg_net_return']:>+8.2f}%")
    print(f"  Max Drawdown            : {results['max_drawdown']:>8.2f}%")

    print("\nTrade Statistics:")
    print(f"  Total Trades    : {results['total_trades']}")
    print(f"  Winning Trades  : {results['winning_trades']}")
    print(f"  Losing Trades   : {results['losing_trades']}")
    print(f"  Win Rate        : {results['win_rate']:.2f}%")

    # ── Pattern breakdown ─────────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("BREAKDOWN BY PATTERN")
    print("-" * 40)
    for pattern in CLASS_NAMES:
        pt = [t for t in all_trades if t.pattern == pattern]
        if pt:
            wins = len([t for t in pt if t.gross_return_pct > 0])
            print(f"\n  {pattern.upper()}:")
            print(f"    Trades    : {len(pt)}")
            print(f"    Win Rate  : {wins / len(pt) * 100:.2f}%")
            print(f"    Avg Gross : {np.mean([t.gross_return_pct for t in pt]):+.2f}%")
            print(f"    Avg Net   : {np.mean([t.net_return_pct   for t in pt]):+.2f}%")

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
