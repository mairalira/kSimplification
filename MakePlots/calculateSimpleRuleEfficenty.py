from typing import List

from models.loadModel import model_batch_classify
from dataSet.load_data import load_dataset_ts

import numpy as np


def calculate_simple_rule_efficenty():
    dataset_name = "Chinatown"
    model_name = "Chinatown_100.keras"

    training_data = load_dataset_ts(dataset_name=dataset_name, data_type="TRAIN")
    test_data = load_dataset_ts(dataset_name=dataset_name, data_type="TEST")
    validation_data = load_dataset_ts(dataset_name=dataset_name, data_type="VALIDATION")

    all_data_combined = np.concatenate((training_data, test_data, validation_data))
    all_classes = model_batch_classify(batch_of_timeseries=all_data_combined, model_name=model_name)

    simple_rule_TBGB = 0
    simple_rule_TBGB_diffs = []
    simple_rule_TBGR = 0
    simple_rule_TBGR_diffs = []
    simple_rule_TRGB = 0
    simple_rule_TRGB_diffs = []
    simple_rule_TRGR = 0
    simple_rule_TRGR_diffs = []
    for ts, c in zip(all_data_combined, all_classes):  # type Lost[float], int
        y_0 = ts[0]
        y_5 = ts[5]
        diff = y_0 - y_5
        simple_rule_guess = None
        if diff >= 200:
            simple_rule_guess = 0
        else:
            simple_rule_guess = 1

        if c == 0 and simple_rule_guess == 0:
            simple_rule_TBGB += 1
            simple_rule_TBGB_diffs.append(diff)
        elif c == 0 and simple_rule_guess == 1:
            simple_rule_TBGR += 1
            simple_rule_TBGR_diffs.append(diff)
        elif c == 1 and simple_rule_guess == 0:
            simple_rule_TRGB += 1
            simple_rule_TRGB_diffs.append(diff)
        elif c == 1 and simple_rule_guess == 1:
            simple_rule_TRGR += 1
            simple_rule_TRGR_diffs.append(diff)
        else:
            raise Exception("Something went wrong")

    print(f"simple_rule_TBGB: {simple_rule_TBGB}. Diff avg {np.average(simple_rule_TBGB_diffs)}")
    print(f"simple_rule_TBGR: {simple_rule_TBGR}. Diff avg {np.average(simple_rule_TBGR_diffs)}")
    print(f"simple_rule_TRGB: {simple_rule_TRGB}. Diff avg {np.average(simple_rule_TRGB_diffs)}")
    print(f"simple_rule_TRGR: {simple_rule_TRGR}. Diff avg {np.average(simple_rule_TRGR_diffs)}")


if __name__ == "__main__":
    calculate_simple_rule_efficenty()
