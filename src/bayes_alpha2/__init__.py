import yfinance as yf
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib


MODEL_PATH = "model.joblib"

# Ask the user for a stock ticker
ticker_input = input("Enter stock ticker symbol (e.g., AAPL, MSFT): ").strip().upper()

# Use the user input in yfinance
stock = yf.Ticker(ticker_input)

historical_stock_data = stock.history(period="5y")

# Extract individual columns as separate variables
open_prices = historical_stock_data['Open']
high_prices = historical_stock_data['High']
low_prices = historical_stock_data['Low']
close_prices = historical_stock_data['Close']
volume_data = historical_stock_data['Volume']

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

# Create target variable (1 if price goes up next day, 0 if down)
def create_target(data):
    """Create target variable: 1 if next day close > today close, 0 otherwise"""
    target = (data['Close'].shift(-1) > data['Close']).astype(int)
    return target

#unsure
#___________________________________________________________

# Generate features and target
features = create_features(historical_stock_data)
target = create_target(historical_stock_data)

# Remove rows with NaN values
features = features.dropna()
target = target[features.index]
target = target.dropna()

# Align features and target
common_index = features.index.intersection(target.index)
features = features.loc[common_index]
target = target.loc[common_index]

#____________________________________________________________

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

# Train Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(nb_classifier, MODEL_PATH)
print("✅ Model saved.")

# Make predictions
y_pred = nb_classifier.predict(X_test)
y_pred_proba = nb_classifier.predict_proba(X_test)

latest_features = features.iloc[-1:]
prediction = nb_classifier.predict(latest_features)[0]
prediction_proba = nb_classifier.predict_proba(latest_features)[0]
print(f"Prediction: {'RISE' if prediction == 1 else 'FALL'}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Training samples: {len(X_train)}")
prediction_date = features.index[-1] + pd.Timedelta(days=1)
print(f"Prediction is for: {prediction_date.date()}")