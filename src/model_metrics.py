import pandas as pd
from functools import wraps
from typing import Any, List
from dataclasses import dataclass
from sklearn.base import BaseEstimator
import time

def with_metrics(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        mm = fn(*args, **kwargs)

        if not mm.train_times or not mm.test_times:
            print("Warning: No timing data available")
            return mm

        # Now metrics.model_train_times is a list, so wrap it in a Series
        avg_train = pd.Series(mm.train_times).mean()
        avg_test  = pd.Series(mm.test_times).mean()

        print(f"Average training time per fold: {avg_train:.4f} seconds")
        print(f"Average testing time per fold : {avg_test:.4f} seconds\n")

        print(f"Total training time: {sum(mm.train_times):.4f} seconds")
        print(f"Total testing time : {sum(mm.test_times):.4f} seconds\n")

        total = sum(mm.train_times) + sum(mm.test_times)
        print(f"Total time: {total:.4f} seconds")
        
        if mm.train_times:
            print(f"Final model training time: {mm.train_times[-1]:.4f} seconds\n")

        for i, score in enumerate(mm.accuracy_scores):
            print(f"Walk-forward accuracy (fold {i+1}): {score:.5f}")

        if mm.accuracy_scores:
            print(f"Overall walk-forward accuracy: {pd.Series(mm.accuracy_scores).mean():.5f}\n")

        if mm.precision is not None:
            print(f"Precision: {mm.precision:.2%}")
        if mm.recall is not None:
            print(f"Recall: {mm.recall:.2%}")
        if mm.f1 is not None:
            print(f"F1 Score: {mm.f1:.2%}")
        if mm.confusion_matrix is not None:
            print(f"Confusion Matrix:\n{mm.confusion_matrix}")

        return mm

    return wrapper

class MethodTimer(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self.train_times = []
        self.test_times = []

    def fit(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = self.estimator.fit(*args, **kwargs)
        self.train_times.append(time.perf_counter() - t0)
        return self

    def predict(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = self.estimator.predict(*args, **kwargs)
        self.test_times.append(time.perf_counter() - t0)
        return result

    def predict_proba(self, *args, **kwargs):
        return self.estimator.predict_proba(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.estimator, attr)

@dataclass
class ModelMetrics:
    prediction: Any
    probability: float
    train_times: List[float]
    test_times: List[float]
    accuracy_scores: List[float]
    model: Any
    precision: float = None
    recall: float = None
    f1: float = None
    confusion_matrix: Any = None