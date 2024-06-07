from typing import List
from simplify.DPcustomAlgoKSmallest import solve_and_find_points
from dataSet.load_data import load_dataset_ts
import random


def get_min_and_max(ts_all):
    """
    returns min_y and max_y in dataset
    :param ts_all:
    :return: min_y, max_y
    """
    max_y = max([max(ts) for ts in ts_all])
    min_y = min([min(ts) for ts in ts_all])
    return min_y, max_y


def dataset_sensitive_c(ts_all, percentage_c):
    min_y, max_y = get_min_and_max(ts_all)

    c = abs(max_y - min_y) * percentage_c
    print(f"c constant {c}")
    return c


def get_time_series_and_seg_pivots(dataset_name, instance_nr=0):
    all_time_series = load_dataset_ts(dataset_name, data_type="TEST")

    # instance_nr = 0
    ts = all_time_series[instance_nr]
    x_values = list(range(len(ts)))
    c_percentage = 200  # Chinatown : 200
    my_k = 1
    my_c = dataset_sensitive_c(all_time_series, c_percentage)  # Chinatown: 1

    random.seed(42)

    all_selected_points, all_ys = solve_and_find_points(X=x_values, Y=ts, c=my_c, K=my_k, saveImg=False)
    best_fit_points = all_selected_points[0]
    best_fit_ys = all_ys[0]

    min_y, max_y = get_min_and_max(all_time_series)
    min_y = min_y - (max_y - min_y) / 4  # Extra buffer
    max_y = max_y + (max_y - min_y) / 4  # Extra buffer

    return best_fit_points, best_fit_ys, ts, min_y, max_y
