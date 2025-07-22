from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import model_metrics
import constants
import splits

def naive_bayes_classifier (X_all, Y_all):
    X_cv, Y_cv, X_live, Y_live = fix_forecast_row(X_all, Y_all)

    pipe = choice_pipe()
    if "multinomial_nb" in constants.registry:    
        # Remove rows with any negative values
        mask = (X_cv >= 0).all(axis=1)
        X_cv = X_cv[mask]
        Y_cv = Y_cv[mask]
        X_live = X_live.clip(lower=0)  # Still clip live data

    timed_pipe = model_metrics.MethodTimer(pipe)

    yt, yp, mw_scores, fitted_pipe = choice_split(X_cv, Y_cv, timed_pipe, n_splits=constants.splits)

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

def fix_forecast_row(X_all, y_all):
    X_live = X_all.iloc[[-1]]
    y_live = y_all.iloc[[-1]]
    X_cv   = X_all.iloc[:-1]
    y_cv   = y_all.iloc[:-1]
    return X_cv, y_cv, X_live, y_live

def choice_split(X, y, pipeline, n_splits):
    match constants.registry:
        case _ if "moving_window_split" in constants.registry:
            return splits.moving_window_split(X, y, pipeline, n_splits)
        case _ if "regular_split" in constants.registry:
            return splits.regular_split(X, y, pipeline)
        case _ if "simple_split" in constants.registry:
            return splits.simple_split(X, y, pipeline)
        case _:
            raise ValueError("No valid split found in registry.")
        
def choice_pipe():
    match constants.registry:
        case _ if "bernoulli_nb" in constants.registry:
            print("Using Bernoulli Naive Bayes")
            return make_pipeline(StandardScaler(), BernoulliNB())
        case _ if "multinomial_nb" in constants.registry:
            print("Using Multinomial Naive Bayes")
            return make_pipeline(StandardScaler(), MultinomialNB())
        case _ if "gaussian_nb" in constants.registry:
            print("Using Gaussian Naive Bayes")
            return make_pipeline(StandardScaler(), GaussianNB())
        case _:
            raise ValueError("No valid Naive Bayes classifier found in registry.")