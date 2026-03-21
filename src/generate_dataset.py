"""
generate_dataset.py
====================
This script downloads stock data, detects candlestick patterns,
and creates images for training a CNN.

Patterns detected:
- Marubozu: large body with very small wicks (strong trend)
- Shooting Star: bearish reversal with long upper wick
"""

import os
import random
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION - Change these values if needed
# ============================================================

# List of stock tickers to download data from
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
    'WMT', 'PG', 'HD', 'DIS', 'NFLX',
    'INTC', 'AMD', 'CSCO', 'PFE', 'KO'
]

# How many candles to include in each image
WINDOW_SIZE = 20

# Image size in pixels
IMAGE_SIZE = 224

# Where to save data
RAW_DATA_DIR = 'data/raw'
IMAGES_DIR = 'data/images'

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# PATTERN DETECTION FUNCTIONS
# ============================================================

def detect_marubozu(candle):
    """
    Detect if a candle is a Marubozu pattern.
    
    A Marubozu has a very large body with minimal wicks.
    This indicates strong buying/selling pressure.
    
    Rules:
    - C > O (bullish candle)
    - upper_wick <= 0.05 * range
    - lower_wick <= 0.05 * range
    - body >= 0.9 * range
    """
    open_price = candle['Open']
    high_price = candle['High']
    low_price = candle['Low']
    close_price = candle['Close']
    
    # Calculate body and range
    body = abs(close_price - open_price)
    candle_range = high_price - low_price
    
    # Avoid division by zero
    if candle_range == 0:
        return False
    
    # Calculate wicks
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    
    # Check all conditions
    is_bullish = close_price > open_price
    has_small_upper_wick = upper_wick <= 0.05 * candle_range
    has_small_lower_wick = lower_wick <= 0.05 * candle_range
    has_large_body = body >= 0.9 * candle_range
    
    is_marubozu = is_bullish and has_small_upper_wick and has_small_lower_wick and has_large_body
    
    return is_marubozu


def detect_shooting_star(candle):
    """
    Detect if a candle is a Shooting Star pattern.
    
    A Shooting Star is a bearish reversal pattern with:
    - Small body at the bottom
    - Long upper wick (shadow)
    - Very small or no lower wick
    
    Rules:
    - Small body: body <= 0.3 * range
    - Long upper wick: upper_wick >= 2 * body
    - Small lower wick: lower_wick <= 0.25 * body
    """
    open_price = candle['Open']
    high_price = candle['High']
    low_price = candle['Low']
    close_price = candle['Close']
    
    # Calculate body and range
    body = abs(close_price - open_price)
    candle_range = high_price - low_price
    
    # Avoid division by zero
    if candle_range == 0:
        return False
    
    # Calculate wicks
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    
    # Check all conditions
    has_small_body = body <= 0.3 * candle_range
    has_long_upper_wick = upper_wick >= 2 * body
    has_small_lower_wick = lower_wick <= 0.25 * body
    
    is_shooting_star = has_small_body and has_long_upper_wick and has_small_lower_wick
    
    return is_shooting_star


def detect_pattern(df, index):
    """
    Detect which pattern (if any) the candle at given index forms.
    
    Priority: shooting_star > marubozu
    
    Returns: 'shooting_star', 'marubozu', or None
    """
    # Get current candle
    current_candle = df.iloc[index]
    
    # Check shooting_star first (highest priority)
    if detect_shooting_star(current_candle):
        return 'shooting_star'
    
    # Check marubozu (second priority)
    if detect_marubozu(current_candle):
        return 'marubozu'
    
    # No pattern found
    return None


# ============================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================

def create_candlestick_image(df_window, save_path):
    """
    Create a candlestick chart image from OHLC data.
    
    Image settings:
    - Size: 224x224 pixels
    - Dark background
    - No axes
    - No volume bars
    """
    # Make sure the DataFrame has a DatetimeIndex (required by mplfinance)
    df_plot = df_window.copy()
    
    # Create custom style with dark background
    custom_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',  # Dark theme
        gridstyle='',  # No grid
    )
    
    # Calculate figure size in inches (224 pixels at 100 DPI)
    fig_size = (IMAGE_SIZE / 100, IMAGE_SIZE / 100)
    
    # Create the candlestick chart
    mpf.plot(
        df_plot,
        type='candle',          # Candlestick chart
        style=custom_style,     # Dark theme
        axisoff=True,           # No axes
        volume=False,           # No volume bars
        savefig=dict(
            fname=save_path,
            dpi=100,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='black'
        ),
        figsize=fig_size
    )


# ============================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================

def download_stock_data(ticker, start_date, end_date):
    """
    Download daily OHLC data for a stock using yfinance.
    
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"  Downloading {ticker}...")
    
    try:
        # Download data
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False  # Don't show download progress bar
        )
        
        # Check if we got data
        if df.empty:
            print(f"  Warning: No data for {ticker}")
            return None
        
        # Handle multi-level columns (yfinance sometimes returns this)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Keep only OHLC columns
        df = df[['Open', 'High', 'Low', 'Close']]
        
        # Remove any rows with missing data
        df = df.dropna()
        
        print(f"  Got {len(df)} days of data for {ticker}")
        return df
        
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        return None


def save_raw_data(df, ticker, output_dir):
    """
    Save the downloaded data to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(filepath)
    print(f"  Saved to {filepath}")


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def generate_samples_from_stock(df, ticker):
    """
    Go through all candles in the stock data and find patterns.
    
    Uses a sliding window approach:
    - Take WINDOW_SIZE candles
    - Check the LAST candle for patterns
    - If pattern found, create sample (window data + label)
    
    Returns a list of samples: [(df_window, pattern_name), ...]
    """
    samples = []
    
    # We need at least WINDOW_SIZE candles
    if len(df) < WINDOW_SIZE:
        return samples
    
    # Slide through the data
    for i in range(WINDOW_SIZE, len(df)):
        # Get the window of candles (last WINDOW_SIZE candles up to position i)
        start_idx = i - WINDOW_SIZE
        end_idx = i
        df_window = df.iloc[start_idx:end_idx].copy()
        
        # The last candle in the window is at index (end_idx - 1) in original df
        last_candle_idx = end_idx - 1
        
        # Detect pattern on the last candle
        pattern = detect_pattern(df, last_candle_idx)
        
        # If a pattern is found, add this sample
        if pattern is not None:
            samples.append((df_window, pattern, ticker, i))
    
    return samples


def split_samples(samples):
    """
    Split samples into train/val/test sets with balanced classes.
    Target: 100-150 images per class per split
    
    Split ratios: 70% train, 15% val, 15% test
    """
    # Group samples by pattern
    pattern_groups = {'marubozu': [], 'shooting_star': []}
    for sample in samples:
        pattern = sample[1]
        pattern_groups[pattern].append(sample)
    
    # Shuffle each group
    for pattern in pattern_groups:
        random.shuffle(pattern_groups[pattern])
    
    # Calculate target counts (aim for 120 train, 32 val, 32 test per class)
    # This gives us ~184 total per class, within 100-150 range per split
    train_samples = []
    val_samples = []
    test_samples = []
    
    for pattern, samples_list in pattern_groups.items():
        n = len(samples_list)
        if n < 184:  # Not enough samples
            # Use what we have with proportional split
            train_end = int(n * 0.70)
            val_end = train_end + int(n * 0.15)
        else:
            # Cap at 184 total: 120 train + 32 val + 32 test
            train_end = 120
            val_end = 120 + 32
            samples_list = samples_list[:184]  # Use only first 184
        
        train_samples.extend(samples_list[:train_end])
        val_samples.extend(samples_list[train_end:val_end])
        test_samples.extend(samples_list[val_end:])
    
    # Shuffle the combined lists
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


def save_samples_as_images(samples, split_name):
    """
    Save a list of samples as candlestick chart images.
    
    Images are saved to: data/images/{split_name}/{pattern_name}/
    """
    print(f"\nSaving {len(samples)} images for {split_name} set...")
    
    # Counter for unique filenames
    counters = {'marubozu': 0, 'shooting_star': 0}
    
    for df_window, pattern, ticker, idx in samples:
        # Create output directory
        output_dir = os.path.join(IMAGES_DIR, split_name, pattern)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename
        counters[pattern] += 1
        filename = f"{pattern}_{ticker}_{counters[pattern]}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Create and save the image
        try:
            create_candlestick_image(df_window, filepath)
        except Exception as e:
            print(f"  Error creating image: {e}")
            continue
    
    # Print summary
    for pattern, count in counters.items():
        print(f"  {pattern}: {count} images")


def main():
    """
    Main function to generate the entire dataset.
    """
    print("=" * 60)
    print("CANDLESTICK PATTERN DATASET GENERATOR")
    print("=" * 60)
    
    # Calculate date range (last 5 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\nDate range: {start_date.date()} to {end_date.date()}")
    print(f"Window size: {WINDOW_SIZE} candles")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} pixels")
    
    # Step 1: Download data for all tickers
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOADING STOCK DATA")
    print("=" * 60)
    
    all_data = {}
    for ticker in TICKERS:
        df = download_stock_data(ticker, start_date, end_date)
        if df is not None and len(df) >= WINDOW_SIZE:
            all_data[ticker] = df
            save_raw_data(df, ticker, RAW_DATA_DIR)
    
    print(f"\nDownloaded data for {len(all_data)} stocks")
    
    # Step 2: Find all patterns
    print("\n" + "=" * 60)
    print("STEP 2: DETECTING PATTERNS")
    print("=" * 60)
    
    all_samples = []
    pattern_counts = {'marubozu': 0, 'shooting_star': 0}
    
    for ticker, df in all_data.items():
        print(f"\nScanning {ticker}...")
        samples = generate_samples_from_stock(df, ticker)
        
        # Count patterns found
        for _, pattern, _, _ in samples:
            pattern_counts[pattern] += 1
        
        all_samples.extend(samples)
        print(f"  Found {len(samples)} patterns")
    
    print(f"\nTotal patterns found:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count}")
    print(f"  TOTAL: {len(all_samples)}")
    
    # Step 3: Split into train/val/test
    print("\n" + "=" * 60)
    print("STEP 3: SPLITTING DATASET")
    print("=" * 60)
    
    train_samples, val_samples, test_samples = split_samples(all_samples)
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Step 4: Generate images (balanced: 100-150 per class)
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING IMAGES (BALANCED)")
    print("=" * 60)
    print("Target: 100-150 images per pattern per split")
    
    save_samples_as_images(train_samples, 'train')
    save_samples_as_images(val_samples, 'val')
    save_samples_as_images(test_samples, 'test')
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nImages saved to: {IMAGES_DIR}/")
    print("Folder structure:")
    print("  train/marubozu/, train/shooting_star/")
    print("  val/marubozu/, val/shooting_star/")
    print("  test/marubozu/, test/shooting_star/")


if __name__ == "__main__":
    main()
