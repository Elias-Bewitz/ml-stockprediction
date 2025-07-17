import pandas as pd

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"  # :contentReference[oaicite:0]{index=0}
stock_symbols = pd.read_csv(url)
SYMBOLS = stock_symbols["Symbol"].tolist()