import pandas as pd
from functools import wraps
from typing import Any, List
from dataclasses import dataclass
from sklearn.base import BaseEstimator
import time
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