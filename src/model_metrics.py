import pandas as pd

def naive_bayes_metrics(model_train_times, model_test_times, final_model_time):
    print(f"Average training time per fold: {pd.Series(model_train_times).mean():.4f} seconds")
    print(f"Average testing time per fold: {pd.Series(model_test_times).mean():.4f} seconds\n")
    print(f"Total training time: {sum(model_train_times):.4f} seconds")
    print(f"Total testing time: {sum(model_test_times):.4f} seconds\n")
    print(f"Total time: {sum(model_train_times) + sum(model_test_times):.4f} seconds")
    print(f"Final model training time: {final_model_time:.4f} seconds\n")
