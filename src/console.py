from model_metrics import naive_bayes_metrics
import constants as proj_consts
import pandas as pd

def result(model_prediction, model_certainty, moving_windows_scores, train_times, test_times, final_train_time):
    naive_bayes_metrics(train_times, test_times, final_train_time)

    prediction_text = f"{proj_consts.GREEN}RISE{proj_consts.RESET}" if model_prediction else f"{proj_consts.RED}FALL{proj_consts.RESET}"
    probability_text = f"{proj_consts.BLUE}{model_certainty:.2%}{proj_consts.RESET}"

    print(f"Walk‑forward accuracy : " f"{proj_consts.BLUE}{pd.Series(moving_windows_scores).mean():.5f}{proj_consts.RESET} ")
    print(f"= {prediction_text} " f"(probability: {probability_text})")