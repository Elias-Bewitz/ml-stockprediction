from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def moving_window_split(X, y, pipeline, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_tr, y_tr)
        pred = pipeline.predict(X_te)
        scores.append(accuracy_score(y_te, pred))
        all_y_true.extend(y.iloc[test_idx])
        all_y_pred.extend(pred)
    return all_y_pred, all_y_true, scores, pipeline

def regular_split(X, y, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=None, shuffle=True, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return (
        list(y_pred),
        list(y_test),
        [acc],
        pipeline
    )

def simple_split(X, y, pipeline, test_frac=0.3):
    n = len(X)
    split_idx = int((1 - test_frac) * n)

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_test  = y.iloc[split_idx:]

    # fit / predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return (
        list(y_pred),
        list(y_test),
        [acc],
        pipeline
    )