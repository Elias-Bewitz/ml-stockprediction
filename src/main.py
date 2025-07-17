import yfinance as yf
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib
import time
import naive_bayes 
import console

SEED: int = 42
MODEL_PATH = "model.joblib"

# Ask the user for a stock ticker
ticker_input = input("Enter stock ticker symbol (e.g., AAPL, MSFT): ").strip().upper()

# Use the user input in yfinance
stock = yf.Ticker(ticker_input)

historical_stock_data = stock.history(period="15y", auto_adjust=True)
historical_stock_data = historical_stock_data.dropna(how='all') # Remove rows with all NaN values
historical_stock_data = historical_stock_data[historical_stock_data['Volume'] > 0]  # Ensure there is volume data

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

X_all   = create_features(historical_stock_data).dropna()
y_all   = (historical_stock_data["Close"].shift(-1) > historical_stock_data["Close"]).astype(int).reindex(X_all.index)

res = naive_bayes.naive_bayes_classifier(X_all, y_all)
console.result(res[0], res[1], res[2], res[3], res[4], res[6], res[7])