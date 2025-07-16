import yfinance as yf
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib

SEED        = 42
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

# Create features for the classifier
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

X_live  = X_all.iloc[[-1]]               # newest timestamp
y_live  = y_all.iloc[[-1]]               # not used, but kept for completeness
X_cv    = X_all.iloc[:-1]                # all but last
y_cv    = y_all.iloc[:-1]

N_SPLITS    = 5  # Number of splits for TimeSeriesSplit
tscv        = TimeSeriesSplit(n_splits=N_SPLITS)
pipe        = make_pipeline(StandardScaler(), GaussianNB())
cv_scores   = []

for train_idx, test_idx in tscv.split(X_cv):
    # shuffle *inside* the training slice only
    train_idx = pd.Series(train_idx).sample(frac=1, random_state=SEED).values
    X_tr, X_te = X_cv.iloc[train_idx], X_cv.iloc[test_idx]
    y_tr, y_te = y_cv.iloc[train_idx], y_cv.iloc[test_idx]

    pipe.fit(X_tr, y_tr)
    cv_scores.append(accuracy_score(y_te, pipe.predict(X_te)))

print(f"Walk‑forward accuracy (mean of {N_SPLITS} folds): "
      f"{pd.Series(cv_scores).mean():.3f}")

pipe.fit(X_cv, y_cv)
joblib.dump(pipe, MODEL_PATH)
print("✅ Final model saved.")

pred_live      = pipe.predict(X_live)[0]
pred_live_prob = pipe.predict_proba(X_live)[0, pred_live]

print(f"Prediction for {X_live.index[0].date() + pd.offsets.BDay(1)} "
      f"= {'RISE' if pred_live else 'FALL'} "
      f"(probability: {pred_live_prob:.2%})")
cm = confusion_matrix(y_cv, pipe.predict(X_cv), labels=[1, 0])
print("Confusion Matrix:")
print(cm)