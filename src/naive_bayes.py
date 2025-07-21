from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import model_metrics

@model_metrics.with_metrics
def naive_bayes_classifier (X_all, Y_all):
    X_cv, Y_cv, X_live, Y_live = fix_forecast_row(X_all, Y_all)

    pipe = make_pipeline(StandardScaler(), MultinomialNB())
    timed_pipe = model_metrics.MethodTimer(pipe)

    yt, yp, mw_scores, fitted_pipe = moving_window_split(X_cv, Y_cv, timed_pipe, n_splits=5)

    timed_pipe.fit(X_cv, Y_cv)

    pred_live = timed_pipe.predict(X_live)[0]
    pred_live_prob = timed_pipe.predict_proba(X_live)[0, pred_live]

    metrics = model_metrics.ModelMetrics(
        prediction=pred_live,
        probability=pred_live_prob,
        train_times=timed_pipe.train_times,
        test_times=timed_pipe.test_times,
        accuracy_scores=mw_scores,
        model=fitted_pipe,
        precision=precision_score(yt, yp),
        recall=recall_score(yt, yp),
        f1=f1_score(yt, yp),
        confusion_matrix=confusion_matrix(yt, yp)
    )
    return metrics

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

def fix_forecast_row(X_all, y_all):
    X_live = X_all.iloc[[-1]]
    y_live = y_all.iloc[[-1]]
    X_cv   = X_all.iloc[:-1]
    y_cv   = y_all.iloc[:-1]
    return X_cv, y_cv, X_live, y_live