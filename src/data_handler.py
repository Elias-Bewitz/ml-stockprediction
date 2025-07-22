import pandas as pd
import yfinance as yf
import numpy as np

def single_stock(ticker):
    stock = yf.Ticker(ticker)
    historical_stock_data = stock.history(period="40y", auto_adjust=True)
    historical_stock_data = historical_stock_data.dropna(how='all') # Remove rows with all NaN values
    historical_stock_data = historical_stock_data[historical_stock_data['Volume'] > 0]  # Ensure there is volume data

    X_all_features   = create_features(historical_stock_data).dropna()
    y_all_targets   = (historical_stock_data["Close"].shift(-1) > historical_stock_data["Close"]).astype(int).reindex(X_all_features.index)

    print(f"Collected {len(X_all_features)} training samples")

    return X_all_features, y_all_targets

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(data):
    """Create features from stock data, including advanced technical indicators."""
    df = pd.DataFrame(index=data.index)
    
    # — Price‐based features
    df['price_change']      = data['Close'].pct_change()
    df['high_low_ratio']    = data['High'] / data['Low']
    df['open_close_ratio']  = data['Open'] / data['Close']
    df['volume_ma_5']       = data['Volume'].rolling(5).mean()
    
    # — Simple moving averages
    df['sma_5']   = data['Close'].rolling(5).mean()
    df['sma_10']  = data['Close'].rolling(10).mean()
    df['price_vs_sma5']  = data['Close'] / df['sma_5']
    df['price_vs_sma10'] = data['Close'] / df['sma_10']
    
    # — Exponential moving averages
    df['ema_5']   = data['Close'].ewm(span=5,  adjust=False).mean()
    df['ema_10']  = data['Close'].ewm(span=10, adjust=False).mean()
    
    # — MACD & Signal line
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    df['macd']         = ema12 - ema26
    df['macd_signal']  = df['macd'].ewm(span=9, adjust=False).mean()
    
    # — Momentum indicators
    df['rsi_14']         = calculate_rsi(data['Close'], window=14)
    df['price_momentum'] = data['Close'] / data['Close'].shift(5)
    
    # — Stochastic Oscillator (%K and %D)
    low14  = data['Low'].rolling(14).min()
    high14 = data['High'].rolling(14).max()
    df['stoch_%K'] = (data['Close'] - low14) / (high14 - low14) * 100
    df['stoch_%D'] = df['stoch_%K'].rolling(3).mean()
    
    # — Williams %R
    df['williams_%R'] = (high14 - data['Close']) / (high14 - low14) * -100
    
    # — On‐Balance Volume (OBV)
    direction = np.sign(data['Close'].diff()).fillna(0)
    df['obv'] = (direction * data['Volume']).cumsum()
    
    # — Money Flow Index (MFI)
    tp  = (data['High'] + data['Low'] + data['Close']) / 3  # Typical Price
    mf  = tp * data['Volume']                              # Money Flow
    # Positive / negative money flows
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfr     = pos_mf / neg_mf
    df['mfi_14'] = 100 - (100 / (1 + mfr))
    
    return df