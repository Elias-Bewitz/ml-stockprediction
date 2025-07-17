from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from model_metrics import naive_bayes_metrics
import constants as proj_consts
import time
import pandas as pd

def naive_bayes_classifier (X_all, Y_all):
    X_live  = X_all.iloc[[-1]]               # newest timestamp
    y_live  = Y_all.iloc[[-1]]               # not used, but kept for completeness
    X_cv    = X_all.iloc[:-1]                # all but last
    y_cv    = Y_all.iloc[:-1]

    pipe        = make_pipeline(StandardScaler(), GaussianNB())

    mw_scores, mw_pipe, train_times, test_times = moving_window_split(X_cv, y_cv, pipe, 5)

    final_train_start = time.time()
    pipe.fit(X_cv, y_cv)
    final_train_time = time.time() - final_train_start

    # Live prediction
    pred_live = pipe.predict(X_live)[0]
    pred_live_prob = pipe.predict_proba(X_live)[0, pred_live]

    return mw_pipe, pred_live, pred_live_prob, mw_scores, train_times, test_times, final_train_time

def moving_window_split(X, y, pipeline, n_splits, random_state=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    train_times = []
    test_times = []
    
    for train_idx, test_idx in tscv.split(X):
        # shuffle within training indices
        shuffled = pd.Series(train_idx).sample(frac=1, random_state=random_state).values
        X_tr, X_te = X.iloc[shuffled], X.iloc[test_idx]
        y_tr, y_te = y.iloc[shuffled], y.iloc[test_idx]
        
        # Time training
        train_start = time.time()
        pipeline.fit(X_tr, y_tr)
        train_time = time.time() - train_start
        train_times.append(train_time)
        
        # Time testing
        test_start = time.time()
        pred = pipeline.predict(X_te)
        test_time = time.time() - test_start
        test_times.append(test_time)
        
        scores.append(accuracy_score(y_te, pred))
    
    return scores, pipeline, train_times, test_times

def regular_split(X, y, pipeline, test_size=0.25, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Time training
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # Time testing
    test_start = time.time()
    pred = pipeline.predict(X_test)
    test_time = time.time() - test_start
    
    score = accuracy_score(y_test, pred)
    return score, pipeline, train_time, test_time