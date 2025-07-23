from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import utils.model_metrics as model_metrics
import constants.constants_injector as c
import train_test_splits.train_test_splits as train_test_splits

def naive_bayes_classifier (X_all, Y_all):
    X_cv, Y_cv, X_live, Y_live = fix_forecast_row(X_all, Y_all)

    pipe = choice_pipe()
    if "multinomial_nb" in c.registry:
        # Remove rows with any negative values
        mask = (X_cv >= 0).all(axis=1)
        X_cv = X_cv[mask]
        Y_cv = Y_cv[mask]
        X_live = X_live.clip(lower=0)  # Still clip live data

    timed_pipe = model_metrics.MethodTimer(pipe)

    yt, yp, mw_scores, fitted_pipe = choice_split(X_cv, Y_cv, timed_pipe, n_splits=c.splits)

    timed_pipe.fit(X_cv, Y_cv)

    pred_live = timed_pipe.predict(X_live)[0]
    print(timed_pipe.predict(X_live)[0])
    print(X_live)
    print(f"Prediction: {'Rise' if pred_live == 1 else 'Fall'}")
    print (pred_live)
    print(f"Date: {X_live.index[0]}")
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
    match c.registry:
        case _ if "moving_window_split" in c.registry:
            return train_test_splits.moving_window_split(X, y, pipeline, n_splits)
        case _ if "regular_split" in c.registry:
            return train_test_splits.regular_split(X, y, pipeline)
        case _ if "simple_split" in c.registry:
            return train_test_splits.simple_split(X, y, pipeline)
        case _:
            raise ValueError("No valid split found in registry.")
        
def choice_pipe():
    match c.registry:
        case _ if "random_forest" in c.registry:
            print("Using Random Forest Classifier")
            from sklearn.ensemble import RandomForestClassifier
            return make_pipeline(StandardScaler(), RandomForestClassifier (
                n_estimators=c.rf_estimators, 
                class_weight=c.rf_class_weight, 
                max_depth=c.rf_max_depth, 
                min_samples_leaf=c.rf_min_samples_leaf, 
                max_features=c.rf_max_features, 
                min_samples_split=c.rf_min_samples_split, 
                random_state=c.rf_random_state)
                )
        case _ if "bernoulli_nb" in c.registry:
            print("Using Bernoulli Naive Bayes")
            return make_pipeline(StandardScaler(), BernoulliNB())
        case _ if "multinomial_nb" in c.registry:
            print("Using Multinomial Naive Bayes")
            return make_pipeline(StandardScaler(), MultinomialNB())
        case _ if "gaussian_nb" in c.registry:
            print("Using Gaussian Naive Bayes")
            return make_pipeline(StandardScaler(), GaussianNB())
        case _:
            raise ValueError("No valid Naive Bayes classifier found in registry.")