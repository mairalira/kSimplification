import random

import numpy as np

from dataSet.load_data import load_dataset_ts
from models.loadModel import batch_confidence, model_batch_classify, model_classify, model_confidence
from simplify.DPcustomAlgoKSmallest import solve_and_find_points
import os


def generate_approximation_ts_for_all_in_dataset():
    all_time_series = load_dataset_ts("Chinatown", data_type="TEST")
    model_name = "Chinatown_1000.keras"
    store_all = False
    # Select one time series
    make_folder("justTS")
    make_folder("bestFit")

    min_y, max_y = get_min_and_max(all_time_series)
    c_percentage = 200

    my_c = dataset_sensitive_c(all_time_series, c_percentage)  # Chinatown: 1
    my_k = 1000
    for ts_nr in [0]:  # range(len(all_time_series)):
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

        # Select segmentations with same classification

        ts_and_class = zip(all_classes, list(range(len(all_interpolations))))

        ts_idx_to_keep = list(map(lambda x: x[1], filter(lambda x: x[0] == org_class, ts_and_class)))
        confidence_of_keep = batch_confidence(model_name=model_name, batch_of_timeseries=list(
            map(lambda x: all_interpolations[x], ts_idx_to_keep)))

        highest_confidence_among_keep_idx = np.argmax(confidence_of_keep)  # np.argmax(confidence_of_keep)
        highest_confidence_idx = ts_idx_to_keep[highest_confidence_among_keep_idx]  # Extract the idx

        class_approx = model_classify(model_name, all_interpolations[highest_confidence_idx])
        confidence_approx = confidence_of_keep[highest_confidence_among_keep_idx]

        from matplotlib import pyplot as plt
        from visualization.plotting import make_plot, TSParam, PlotParams

        tsParams = TSParam(x_values=x_values, y_values=ts, fmat='x', color='black')
        save_file = f'img/justTS/{ts_nr}.png'
        x_lim = (-1, 24)
        y_lim = (min_y - abs(max_y - min_y) * 0.1, max_y + abs(max_y - min_y) * 0.1)
        plotParams = PlotParams(ts_params=[tsParams], save_file=save_file, x_lim=x_lim, y_lim=y_lim)
        make_plot(plotParams)

        tsParamOrg = TSParam(x_values=x_values, y_values=ts, fmat='x', color='black')
        tsParamSeg = TSParam(x_values=x_values, y_values=all_interpolations[highest_confidence_idx],
                             fmat='--o', color='blue')
        tsParamConfidence = TSParam(x_values=all_selected_points[highest_confidence_idx],
                                    y_values=all_ys[highest_confidence_idx],
                                    fmat='o', color='red')
        x_lim = (-1, 24)
        y_lim = (min_y - abs(max_y - min_y) * 0.1, max_y + abs(max_y - min_y) * 0.1)
        title = f"Class org:{org_class}  Confidence org:{org_confidence:.2f}, class_approx:{class_approx} Confidence_approx:{confidence_approx:.2f}"
        file_name = f'img/bestFit/{ts_nr}.png'
        plotParams = PlotParams(ts_params=[tsParamOrg, tsParamSeg, tsParamConfidence], title=title, x_lim=x_lim,
                                y_lim=y_lim,
                                save_file=file_name)
        make_plot(plotParams)

        # plt.clf()
        # plt.xlim(-1, 24)
        # plt.ylim(min_y - abs(max_y - min_y) * 0.1, max_y + abs(max_y - min_y) * 0.1)
        # plt.plot(x_values, ts, 'x', color='black')
        # plt.plot(x_values, all_interpolations[highest_confidence_idx], '--o', color='blue')
        # plt.plot(all_selected_points[highest_confidence_idx], all_ys[highest_confidence_idx], 'o', color='red')
        # plt.title(f"Class org:{org_class}  Confidence org:{org_confidence:.2f}, class_approx:{class_approx} Confidence_approx:{confidence_approx:.2f}")
        # plt.savefig(f'img/bestFit/{ts_nr}.png')


if __name__ == '__main__':
    generate_approximation_ts_for_all_in_dataset()
