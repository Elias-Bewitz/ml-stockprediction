import yfinance as yf

stock = yf.Ticker("AAPL")

historical_stock_data = stock.history(period="1y")

with open("historical_data.txt", "w") as file:
    file.write(historical_stock_data.to_string())