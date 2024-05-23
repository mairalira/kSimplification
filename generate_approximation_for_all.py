import random

import numpy as np

from dataSet.load_data import load_dataset_ts
from models.loadModel import batch_confidence, model_batch_classify, model_classify, model_confidence
from simplify.DPcustomAlgoKSmallest import solve_and_find_points
import os


def make_folder(folderName):
    # Create the new folder
    parent_folder = "img/"
    relative_folder_location = parent_folder + folderName
    os.makedirs(relative_folder_location, exist_ok=True)


def calculate_line_equation(x1, y1, x2, y2, x3):
    # Calculate the slope (m)
    delta_x = x2 - x1
    if delta_x == 0:
        raise ValueError("The points must have different x-coordinates to calculate the slope.")
    m = (y2 - y1) / delta_x

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y3 at x3
    y3 = m * x3 + b

    return y3


def interpolate_points_to_line(ts_lenght, x_selcted, y_selcted):
    """
    Given a list (points) of [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] of selected points calculate the y value of
    each timeStep.

    For each x in range(timeStep) we have 3 cases:
    1. x1 <= x <= x4: Find the pair xi <= x <=xi+1, s.t. i<=3. Use this slope to find the corresponding y value.
    2. x < x1. Extend the slope between x1 and x2 to x, and find the corresponding y value.
    3. x4 < x. Extend the slope between x3 and x4 to x, and find the corresponding y value.
    :param length: Length of time series
    :return:
    """

    inter_ts = [0 for _ in range(ts_lenght)]
    pointsX = 0
    for i, x in enumerate(range(ts_lenght)):
        if pointsX < len(x_selcted) - 2 and x > x_selcted[pointsX + 1]:
            pointsX += 1

        x1 = x_selcted[pointsX]
        x2 = x_selcted[pointsX + 1]
        y1 = y_selcted[pointsX]
        y2 = y_selcted[pointsX + 1]
        x3 = x
        y3 = calculate_line_equation(x1, y1, x2, y2, x3)
        inter_ts[x] = y3

    return inter_ts


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
    return c


def generate_approximation_ts_for_all_in_dataset():
    all_time_series = load_dataset_ts("Chinatown", data_type="TEST")
    model_name = "Chinatown_1000.keras"
    store_all = True
    # Select one time series
    make_folder("justTS")
    make_folder("bestFit")

    min_y, max_y = get_min_and_max(all_time_series)
    c_percentage = 200

    my_c = dataset_sensitive_c(all_time_series, c_percentage)  # Chinatown: 1
    my_k = 1000
    for ts_nr in range(len(all_time_series)):
        print(ts_nr)
        ts = all_time_series[ts_nr]
        print(f"TS: {ts}")

        x_values = [i for i in range(len(ts))]

        all_selected_points, all_ys = solve_and_find_points(x_values, ts, my_c, my_k, saveImg=False)
        all_interpolations = []
        if store_all:
            make_folder(str(ts_nr))
        for i, (selected_points, ys) in enumerate(zip(all_selected_points, all_ys)):
            inter_ts = interpolate_points_to_line(ts_lenght=len(ts), x_selcted=selected_points, y_selcted=ys)
            all_interpolations.append(inter_ts)

        org_class = model_classify(model_name, ts)
        org_confidence = model_confidence(model_name, ts)
        all_classes = model_batch_classify(model_name, all_interpolations)
        all_confidence = batch_confidence(model_name, all_interpolations)
        if store_all:
            for i, (inter_ts, selected_points, ys) in enumerate(zip(all_interpolations, all_selected_points, all_ys)):
                from matplotlib import pyplot as plt
                plt.clf()
                plt.plot(x_values, ts, 'x', color='black')
                plt.plot(x_values, inter_ts, '--o', color='blue')
                plt.plot(selected_points, ys, '--D', color='red')
                plt.title(
                    f"Org confidence: {org_confidence:.2f}, Org class: {org_class} Curr class: {all_classes[i]} Curr confidence: {all_confidence[i]:.2f}")
                plt.savefig(f'img/{ts_nr}/{i}.png')
        ts_and_class = zip(all_classes, list(range(len(all_interpolations))))

        ts_idx_to_keep = list(map(lambda x: x[1], filter(lambda x: x[0] == org_class, ts_and_class)))
        confidence_of_keep = batch_confidence(model_name=model_name, batch_of_timeseries=list(
            map(lambda x: all_interpolations[x], ts_idx_to_keep)))

        highest_confidence_among_keep_idx = np.argmax(confidence_of_keep)  # np.argmax(confidence_of_keep)
        highest_confidence_idx = ts_idx_to_keep[highest_confidence_among_keep_idx]  # Extract the idx

        class_approx = model_classify(model_name, all_interpolations[highest_confidence_idx])
        confidence_approx = confidence_of_keep[highest_confidence_among_keep_idx]

        from matplotlib import pyplot as plt
        plt.clf()
        # Make test img
        plt.xlim(-1, 24)
        plt.ylim(min_y - abs(max_y - min_y) * 0.1, max_y + abs(max_y - min_y) * 0.1)
        plt.plot(x_values, ts, 'x', color='black')
        plt.savefig(f'img/justTS/{ts_nr}.png')

        plt.clf()
        plt.xlim(-1, 24)
        plt.ylim(min_y - abs(max_y - min_y) * 0.1, max_y + abs(max_y - min_y) * 0.1)
        plt.plot(x_values, ts, 'x', color='black')
        plt.plot(x_values, all_interpolations[highest_confidence_idx], '--o', color='blue')
        plt.plot(all_selected_points[highest_confidence_idx], all_ys[highest_confidence_idx], 'o', color='red')
        plt.title(
            f"Class org:{org_class}  Confidence org:{org_confidence:.2f}, class_approx:{class_approx} Confidence_approx:{confidence_approx:.2f}")
        plt.savefig(f'img/bestFit/{ts_nr}.png')


if __name__ == '__main__':
    generate_approximation_ts_for_all_in_dataset()
