import yfinance as yf
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib

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

def moving_window_split(X, y, pipeline, n_splits, random_state=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        # shuffle within training indices
        shuffled = pd.Series(train_idx).sample(frac=1, random_state=random_state).values
        X_tr, X_te = X.iloc[shuffled], X.iloc[test_idx]
        y_tr, y_te = y.iloc[shuffled], y.iloc[test_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_te, pipeline.predict(X_te)))
    return scores, pipeline

def regular_split(X, y, pipeline, test_size=0.25, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    score = accuracy_score(y_test, pipeline.predict(X_test))
    return score, pipeline

X_all   = create_features(historical_stock_data).dropna()
y_all   = (historical_stock_data["Close"].shift(-1) > historical_stock_data["Close"]).astype(int).reindex(X_all.index)

X_live  = X_all.iloc[[-1]]               # newest timestamp
y_live  = y_all.iloc[[-1]]               # not used, but kept for completeness
X_cv    = X_all.iloc[:-1]                # all but last
y_cv    = y_all.iloc[:-1]

pipe        = make_pipeline(StandardScaler(), GaussianNB())
cv_scores   = []

user_chosen_split = input("Choose split method (1 for regular, 2 for moving window): ")
if user_chosen_split == "1":
    rs_scores, rs_pipe = regular_split(X_cv, y_cv, pipe)
    print(f"Regular split accuracy : {rs_scores:.5f} ")
else:
    mw_scores, mw_pipe = moving_window_split(X_cv, y_cv, pipe, 5)
    print(f"Walk‑forward accuracy : "
          f"{pd.Series(mw_scores).mean():.5f} ")

pipe.fit(X_cv, y_cv)
pred_live      = pipe.predict(X_live)[0]
pred_live_prob = pipe.predict_proba(X_live)[0, pred_live]

print(f"Prediction for {X_live.index[0].date() + pd.offsets.BDay(1)} "
      f"= {'RISE' if pred_live else 'FALL'} "
      f"(probability: {pred_live_prob:.2%})")