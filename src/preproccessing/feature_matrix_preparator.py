import pandas as pd
import yfinance as yf
import numpy as np
import constants.constants_injector as constants_injector

def single_stock(ticker, prediction_horizon='1d'):
    stock = yf.Ticker(ticker)
    historical_stock_data = stock.history(period=constants_injector.year, auto_adjust=constants_injector.auto_adjust)
    historical_stock_data = historical_stock_data.dropna(how='all')
    historical_stock_data = historical_stock_data[historical_stock_data['Volume'] > 0]
    print(f"Historical stock data shape: {historical_stock_data.shape}")

    X_all_features = create_features(historical_stock_data, prediction_horizon).dropna()
    y_all_targets = create_targets(historical_stock_data, prediction_horizon).reindex(X_all_features.index)

    horizon_name = {'1d': 'daily', '1w': 'weekly', '1m': 'monthly'}[prediction_horizon]
    print(f"Collected {len(X_all_features)} training samples for {horizon_name} prediction")

    return X_all_features, y_all_targets

def create_targets(data, horizon='1d', threshold=0.02):
    if horizon == '1d':
        future_returns = data['Close'].pct_change().shift(-1)
    elif horizon == '1w':
        future_returns = (data['Close'].shift(-5) / data['Close'] - 1)
    elif horizon == '1m':
        future_returns = (data['Close'].shift(-22) / data['Close'] - 1)
    else:
        raise ValueError("horizon must be '1d', '1w', or '1m'")
    
    thresholds = {'1d': 0.01, '1w': 0.03, '1m': 0.05}
    threshold = thresholds[horizon]
    
    return (future_returns > threshold).astype(int)

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(data, horizon='1d'):
    df = pd.DataFrame(index=data.index)
    
    df['price_change'] = data['Close'].pct_change().shift(1)
    df['high_low_ratio'] = (data['High'] / data['Low']).shift(1)
    df['open_close_ratio'] = (data['Open'] / data['Close']).shift(1)
    
    if horizon == '1d':
        df['sma_5'] = data['Close'].rolling(5).mean().shift(1)
        df['sma_10'] = data['Close'].rolling(10).mean().shift(1)
        df['rsi_14'] = calculate_rsi(data['Close'], window=14).shift(1)
        df['volume_ratio'] = (data['Volume'] / data['Volume'].rolling(5).mean()).shift(1)
        
    elif horizon == '1w':
        df['sma_10'] = data['Close'].rolling(10).mean().shift(1)
        df['sma_20'] = data['Close'].rolling(20).mean().shift(1)
        df['rsi_14'] = calculate_rsi(data['Close'], window=14).shift(1)
        df['volume_ratio'] = (data['Volume'] / data['Volume'].rolling(10).mean()).shift(1)
        df['weekly_volatility'] = data['Close'].pct_change().rolling(5).std().shift(1)
        
    elif horizon == '1m':
        df['sma_20'] = data['Close'].rolling(20).mean().shift(1)
        df['sma_50'] = data['Close'].rolling(50).mean().shift(1)
        df['sma_200'] = data['Close'].rolling(200).mean().shift(1)
        df['rsi_14'] = calculate_rsi(data['Close'], window=14).shift(1)
        df['monthly_momentum'] = (data['Close'] / data['Close'].shift(22)).shift(1)
        df['monthly_volatility'] = data['Close'].pct_change().rolling(22).std().shift(1)
        df['volume_trend'] = (data['Volume'].rolling(22).mean() / data['Volume'].rolling(66).mean()).shift(1)
        
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26).shift(1)
    df['macd_signal']  = df['macd'].ewm(span=9, adjust=False).mean()
    
    df['rsi_14']         = calculate_rsi(data['Close'], window=14)
    df['price_momentum'] = data['Close'] / data['Close'].shift(5)
    
    low14  = data['Low'].rolling(14).min()
    high14 = data['High'].rolling(14).max()
    df['stoch_%K'] = (data['Close'] - low14) / (high14 - low14) * 100
    df['stoch_%D'] = df['stoch_%K'].rolling(3).mean()
    
    df['williams_%R'] = (high14 - data['Close']) / (high14 - low14) * -100
    
    direction = np.sign(data['Close'].diff()).fillna(0)
    df['obv'] = (direction * data['Volume']).cumsum()
    
    tp  = (data['High'] + data['Low'] + data['Close']) / 3
    mf  = tp * data['Volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfr     = pos_mf / neg_mf
    df['mfi_14'] = 100 - (100 / (1 + mfr))
    
    return df
