from typing import List


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
