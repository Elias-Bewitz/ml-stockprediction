import constants as proj_consts
import pandas as pd
import yfinance as yf

def single_stock(ticker):

    stock = yf.Ticker(ticker)
    historical_stock_data = stock.history(period="15y", auto_adjust=True)
    historical_stock_data = historical_stock_data.dropna(how='all') # Remove rows with all NaN values
    historical_stock_data = historical_stock_data[historical_stock_data['Volume'] > 0]  # Ensure there is volume data

    X_all_features   = create_features(historical_stock_data).dropna()
    y_all_targets   = (historical_stock_data["Close"].shift(-1) > historical_stock_data["Close"]).astype(int).reindex(X_all_features.index)

    print(f"Collected {len(X_all_features)} training samples")

    return X_all_features, y_all_targets

def multi_stock():
    pass

def fetch_stock_data(symbol, period="3y"):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, auto_adjust=True)
        if data.empty or len(data) < 50:  # Skip if insufficient data
            return None
        data = data.dropna(how='all')
        data = data[data['Volume'] > 0]
        return data
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def collect_training_data(symbols, period="3y"):
    all_features = []
    all_targets = []

    for i, symbol in enumerate(symbols):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing {i+1}/{len(symbols)} stocks...")
        
        data = fetch_stock_data(symbol, period)
        if data is None:
            continue
            
        features = create_features(data).dropna()
        if len(features) < 20:  # Skip if insufficient features
            continue
            
        # Create target (next day price movement)
        target = (data["Close"].shift(-1) > data["Close"]).astype(int).reindex(features.index)
        
        # Add stock symbol as a feature (encoded as hash for simplicity)
        features['stock_hash'] = hash(symbol) % 1000  # Simple encoding
        
        all_features.append(features)
        all_targets.append(target)

    # Combine all data
    if all_features:
        X_combined = pd.concat(all_features, ignore_index=True)
        y_combined = pd.concat(all_targets, ignore_index=True)
        return X_combined, y_combined
    else:
        raise ValueError("No valid stock data found")
    
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_features(data):
    """Create features from stock data"""
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['price_change'] = data['Close'].pct_change()  # Daily return
    features['high_low_ratio'] = data['High'] / data['Low']  # Volatility indicator
    features['open_close_ratio'] = data['Open'] / data['Close']  # Intraday movement
    features['volume_ma'] = data['Volume'].rolling(window=5).mean()  # 5-day volume average
    
    # Technical indicators
    features['sma_5'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
    features['sma_10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
    features['price_vs_sma5'] = data['Close'] / features['sma_5']  # Price relative to SMA
    features['price_vs_sma10'] = data['Close'] / features['sma_10']  # Price relative to SMA
    
    # Momentum indicators
    features['rsi'] = calculate_rsi(data['Close'], window=14)
    features['price_momentum'] = data['Close'] / data['Close'].shift(5)  # 5-day momentum
    
    return features