"""
Bitcoin Price Data Preprocessing Automation Script
Author: [Your Name]
Date: 2025-12-17

This script automates the downloading and preprocessing of Bitcoin price data
for the machine learning pipeline. It can be run daily via GitHub Actions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: pandas Series of price data
        window: lookback period (default: 14)
    
    Returns:
        pandas Series of RSI values
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def download_btc_data(start_date='2019-01-01'):
    """
    Download Bitcoin historical price data from Yahoo Finance
    
    Args:
        start_date: Start date for data download (format: 'YYYY-MM-DD')
    
    Returns:
        pandas DataFrame with BTC price data
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Downloading Bitcoin data: {start_date} to {end_date}")
    
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    
    if btc_data.empty:
        raise ValueError("Failed to download Bitcoin data")
    
    print(f"Downloaded {len(btc_data)} records")
    return btc_data


def create_technical_indicators(df):
    """
    Create technical indicators for feature engineering
    
    Args:
        df: pandas DataFrame with OHLCV data
    
    Returns:
        pandas DataFrame with additional technical indicators
    """
    print("Creating technical indicators...")
    
    # Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Price and Volume Changes
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    
    # High-Low Spread
    df['HL_Spread'] = df['High'] - df['Low']
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    print("Technical indicators created successfully")
    return df


def create_target_variable(df):
    """
    Create binary target variable for classification
    
    Args:
        df: pandas DataFrame with price data
    
    Returns:
        pandas DataFrame with target variable
    """
    # Target: 1 if price goes up tomorrow, 0 if goes down or stays same
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df


def preprocess_btc_data(start_date='2019-01-01', 
                       raw_output='btc_raw.csv',
                       processed_output='btc_preprocessing.csv'):
    """
    Main preprocessing pipeline
    
    Args:
        start_date: Start date for data download
        raw_output: Filename for raw data CSV
        processed_output: Filename for preprocessed data CSV
    
    Returns:
        pandas DataFrame with preprocessed data
    """
    try:
        # Step 1: Download data
        btc_data = download_btc_data(start_date)
        
        # Step 2: Save raw data
        btc_data.to_csv(raw_output)
        print(f"Raw data saved: {raw_output}")
        
        # Step 3: Create technical indicators
        df = btc_data.copy()
        df = create_technical_indicators(df)
        
        # Step 4: Create target variable
        df = create_target_variable(df)
        
        # Step 5: Remove missing values
        initial_rows = len(df)
        df_clean = df.dropna()
        removed_rows = initial_rows - len(df_clean)
        print(f"Removed {removed_rows} rows with missing values")
        
        # Step 6: Select features for final dataset
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'MA_7', 'MA_30', 'RSI', 'Price_Change', 
                   'Volume_Change', 'HL_Spread', 'MACD', 'Target']
        
        preprocessed_df = df_clean[features]
        
        # Step 7: Save preprocessed data
        preprocessed_df.to_csv(processed_output)
        print(f"Preprocessed data saved: {processed_output}")
        
        # Step 8: Print summary
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Total records: {len(preprocessed_df)}")
        print(f"Features: {len(features) - 1}")  # Exclude Target
        print(f"Date range: {preprocessed_df.index[0].strftime('%Y-%m-%d')} to {preprocessed_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Target distribution:")
        print(f"  - TURUN (0): {(preprocessed_df['Target'] == 0).sum()}")
        print(f"  - NAIK (1): {(preprocessed_df['Target'] == 1).sum()}")
        print("="*50)
        
        return preprocessed_df
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    print("="*50)
    print("BITCOIN DATA PREPROCESSING AUTOMATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # Run preprocessing
    df = preprocess_btc_data()
    
    print("\nPreprocessing completed successfully!")
