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
import constants as proj_consts
from data_handler import single_stock

ticker_input = input("Enter stock ticker symbol (e.g., AAPL, MSFT): ").strip().upper()

X_all, y_all = single_stock(ticker_input)

res = naive_bayes.naive_bayes_classifier(X_all, y_all)