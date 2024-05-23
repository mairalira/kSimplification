import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import random
from models.loadModel import model_batch_classify, model_classify
from generate_approximation_for_all import dataset_sensitive_c, get_min_and_max, interpolate_points_to_line
from dataSet.load_data import load_dataset_ts
from simplify.DPcustomAlgoKSmallest import solve_and_find_points


def to_full_lines(points_x, all_points_y, ts_length):
    # Convert every perturbed set of points to lines
    line_time_series = {}
    for idx_x in all_points_y.keys():
        line_time_series[idx_x] = {}
        for y_test in all_points_y[idx_x].keys():
            line_version = interpolate_points_to_line(ts_lenght=ts_length, x_selcted=points_x,
                                                      y_selcted=all_points_y[idx_x][y_test])
            line_time_series[idx_x][y_test] = line_version
    return line_time_series


def _create_perturbations_by_single_x_point(points_y, min_y, max_y) -> dict:
    resolution = 10 ** 4
    all_test_ys = list(np.linspace(min_y, max_y, resolution, endpoint=True))
    new_y_values_for_x_points = {}
    for i in range(len(points_y)):
        change_curr_x_pos = _change_of_specific_x_value(nr_x=i, points_y=points_y, all_test_ys=all_test_ys)
        new_y_values_for_x_points[i] = change_curr_x_pos
    return new_y_values_for_x_points


def _change_of_specific_x_value(nr_x, points_y, all_test_ys) -> dict:
    change_curr_x_pos = {}
    for new_y in all_test_ys:
        new_pos_vals = points_y[:nr_x] + [new_y] + points_y[nr_x + 1:]
        change_curr_x_pos[new_y] = new_pos_vals
    return change_curr_x_pos


def _get_feature_attribution_points(ts_length, points_x, points_y):
    return None


def get_time_series(instance_nr=0):
    dataset_name = "Chinatown"
    all_time_series = load_dataset_ts(dataset_name, data_type="TEST")

    # instance_nr = 0
    ts = all_time_series[instance_nr]
    x_values = list(range(len(ts)))
    c_percentage = 200
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


def test(instance_nr):
    best_fit_points, best_fit_ys, ts, min_y, max_y = get_time_series(instance_nr)
    best_fit_ys_org = best_fit_ys.copy()
    best_fit_points_org = best_fit_points.copy()
    approx_line = interpolate_points_to_line(len(ts), best_fit_points, best_fit_ys)
    best_fit_points[0] = 0
    best_fit_points[-1] = len(ts) - 1

    best_fit_ys[0] = approx_line[0]
    best_fit_ys[-1] = approx_line[-1]

    new_ys = _create_perturbations_by_single_x_point(points_y=best_fit_ys, min_y=min_y, max_y=max_y)
    for ys in new_ys:
        print(ys)
    line_versions_dict = to_full_lines(points_x=best_fit_points, all_points_y=new_ys, ts_length=len(ts))
    dict_class = {}
    x_y_c = []
    for x_idx in line_versions_dict.keys():
        dict_class[x_idx] = {}
        for y_value in line_versions_dict[x_idx].keys():
            curr_line = line_versions_dict[x_idx][y_value]
            class_curr = model_classify(model_name="Chinatown_1000.keras", time_series=curr_line)
            dict_class[x_idx][y_value] = class_curr
            x_y_c.append([x_idx, y_value, class_curr])

    # classified = model_batch_classify(model_name="Chinatown_1000.keras", batch_of_timeseries=line_versions)
    # unique, counts = np.unique(classified, return_counts=True)
    # print(unique, counts)
    print(x_y_c)
    x_y_c = np.array(x_y_c)
    class_0 = x_y_c[list(map(lambda x: x[2] == 0, x_y_c))]
    class_1 = x_y_c[list(map(lambda x: x[2] == 1, x_y_c))]
    X_0 = [best_fit_points[int(x)] for x, y, c in class_0]
    Y_0 = [y for x, y, c in class_0]

    X_1 = [best_fit_points[int(x)] for x, y, c in class_1]
    Y_1 = [y for x, y, c in class_1]
    fig, ax = plt.subplots()
    plt.scatter(X_0, Y_0, color=(0, 0, 1, 0.01))
    plt.scatter(X_1, Y_1, color=(1, 0, 0, 0.01))
    plt.plot(list(range(len(ts))), ts, color='grey')
    plt.plot(list(range(len(approx_line))), approx_line, "--", color='black')
    line_above = approx_line + (max_y - min_y) / 10
    line_below = approx_line - (max_y - min_y) / 10

    plt.title(f"Feature Attribution instance nr: {instance_nr}")

    # Make it look nice
    plt.plot(list(range(len(line_above))), line_above, color='black')
    plt.plot(list(range(len(line_below))), line_below, color='black')
    y_lim_max = max_y + (max_y - min_y) / 10
    y_lim_min = min_y - (max_y - min_y) / 10
    plt.ylim((y_lim_min, y_lim_max))
    plt.xlim((-1, 24))
    for i, x in enumerate(best_fit_points_org):
        y = best_fit_ys_org[i]
        x_axsis_r = 0.4
        y_axsis_r = x_axsis_r * (y_lim_max - y_lim_min) / (24 - (-1))
        ellipse = patches.Ellipse((x, y), x_axsis_r, y_axsis_r, fill=True, color='grey')
        ax.add_patch(ellipse)

    for i, x in enumerate(best_fit_points):
        y = best_fit_ys[i]
        x_axsis_r = 0.6
        y_axsis_r = x_axsis_r * (y_lim_max - y_lim_min) / (24 - (-1))
        ellipse = patches.Ellipse((x, y), x_axsis_r, y_axsis_r, fill=False, color='black')
        ax.add_patch(ellipse)

    plt.show()


if __name__ == '__main__':
    for i in range(0, 100, 10):
        test(i)
