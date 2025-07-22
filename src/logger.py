from typing import Any, List
import logging
import pandas as pd
from colorama import init, Fore, Style
import model_metrics

# initialize colorama (on Windows this enables ANSI support)
init(autoreset=True)

# 1) configure a logger
logger = logging.getLogger("model_metrics")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
logger.addHandler(handler)

def log_metrics(mm: model_metrics.ModelMetrics):
    # compute aggregates
    avg_train = pd.Series(mm.train_times).mean()
    avg_test  = pd.Series(mm.test_times).mean()
    total_train = sum(mm.train_times)
    total_test  = sum(mm.test_times)
    total_time  = total_train + total_test

    # 2) log with colors
    logger.info(f"Avg train time/fold: {Fore.YELLOW}{avg_train:.4f}{Style.RESET_ALL}s")
    logger.info(f"Avg test  time/fold: {Fore.YELLOW}{avg_test:.4f}{Style.RESET_ALL}s\n")

    logger.info(f"Total train time:      {Fore.CYAN}{total_train:.4f}{Style.RESET_ALL}s")
    logger.info(f"Total test  time:      {Fore.CYAN}{total_test:.4f}{Style.RESET_ALL}s\n")

    logger.info(f"Overall runtime:       {Fore.MAGENTA}{total_time:.4f}{Style.RESET_ALL}s")
    if mm.train_times:
        logger.info(f"Last fold train time:  {Fore.MAGENTA}{mm.train_times[-1]:.4f}{Style.RESET_ALL}s\n")

    # per-fold accuracies
    for i, score in enumerate(mm.accuracy_scores, start=1):
        logger.info(f"Fold {i} accuracy:      {Fore.GREEN}{score:.5f}{Style.RESET_ALL}")
    if mm.accuracy_scores:
        overall_acc = pd.Series(mm.accuracy_scores).mean()
        logger.info(f"Mean accuracy:         {Fore.LIGHTRED_EX}{overall_acc:.5f}{Style.RESET_ALL}\n")

    # classification metrics
    if mm.precision is not None:
        logger.info(f"Precision:             {Fore.BLUE}{mm.precision:.2%}{Style.RESET_ALL}")
    if mm.recall is not None:
        logger.info(f"Recall:                {Fore.BLUE}{mm.recall:.2%}{Style.RESET_ALL}")
    if mm.f1 is not None:
        logger.info(f"F1 Score:              {Fore.BLUE}{mm.f1:.2%}{Style.RESET_ALL}")
    if mm.confusion_matrix is not None:
        logger.info(f"Confusion Matrix:\n{Fore.WHITE}{mm.confusion_matrix}{Style.RESET_ALL}")