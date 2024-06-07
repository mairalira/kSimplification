import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import os

import numpy as np
import random
from typing import List, Dict, Tuple
from collections import defaultdict

from models.loadModel import model_batch_classify, model_classify
from utils.data import dataset_sensitive_c, get_min_and_max, get_time_series_and_seg_pivots
from utils.line import interpolate_points_to_line, convert_all_points_to_lines
from dataSet.load_data import load_dataset_ts

from utils.model import class_to_color
from visualization.plotting import ScatterParams, PlotParams
from visualization.getTSParam import get_ts_param_org, get_ts_param_approx
from visualization.getEllipseParam import make_all_ellipse_param


class PerturbationTS:
    new_x: int
    new_y: float
    idx_pivots: int

    x_pivots: List[int]
    y_pivots: List[float]

    line_version: List[float]

    pred_class: int

    def __init__(self, new_x: int, new_y: float, idx_pivots: int, x_pivots: List[int], y_pivots: List[float]):
        self.new_x = new_x
        self.new_y = new_y

        self.idx_pivots = idx_pivots
        self.x_pivots = x_pivots
        self.y_pivots = y_pivots

    def set_class(self, pred_class: int):
        self.pred_class = pred_class

    def set_line_version(self, ts_length: int):
        line_version = interpolate_points_to_line(ts_length=ts_length, x_selected=self.x_pivots,
                                                  y_selected=self.y_pivots)
        self.line_version = line_version


def create_x_y_perturbation(org_pivots_y: List[float], org_pivots_x: List[int], ts_length: int, epsilon: float) -> \
        List[PerturbationTS]:
    resolution = 10 ** 3
    all_ys_perturbations = list(np.linspace(-epsilon, epsilon, resolution, endpoint=True))
    x_change = 1
    num_changes = x_change * 2 + 1
    all_x_perturbations = list(np.linspace(-x_change, x_change, num_changes, endpoint=True))

    all_perturbations = []
    for idx in range(len(org_pivots_x)):
        for x_change in all_x_perturbations:
            if x_change != 0 and (x_change + org_pivots_x[idx] in org_pivots_x or x_change + org_pivots_x[idx] < 0
                                  or x_change + org_pivots_x[idx] >= ts_length):
                continue
                # We don't want to change x so much it overlaps with another x in org_pivots_x
                # NB: We can still have that two perturbations overlap
                # NB: Don't change x out of the range [0 .. ts_length-1]
            for y_change in all_ys_perturbations:
                new_y_value = org_pivots_y[idx] + y_change
                new_x_value = org_pivots_x[idx] + x_change
                new_y_pivots = org_pivots_y[:idx] + [new_y_value] + org_pivots_y[idx + 1:]
                new_x_pivots = org_pivots_x[:idx] + [new_x_value] + org_pivots_x[idx + 1:]
                new_perturbation = PerturbationTS(new_x=new_x_value, new_y=new_y_value, idx_pivots=idx,
                                                  x_pivots=new_x_pivots, y_pivots=new_y_pivots)
                all_perturbations.append(new_perturbation)

    return all_perturbations


def make_perturbations(pivots_y_original: List[float], pivots_x_original: List[int],
                       line_version_original: List[float], epsilon: float) -> List[PerturbationTS]:
    """
    We define a perturbation to be a combination of two changes.
    1. Changing a y value in pivots_y. I.e. pivots_y = [0,0,2,2,1] -> [0,0,4,2,2,1]
    2. Changing an x value in pivots_x. I.e. pivots_x = [0,2,5,8,10] -> [0,2,3,8,10]

    To identify a perturbation we will need a tuple of 3 items:
    Index of pivot to change, x changed of pivot, y changed of pivot.

    example:
    Og.piv_x = [0,5,10], Og.piv_y = [10,5,10]
    perturbation: change middle higher and more right->
    Per.piv_x = [0,7,10], Per.piv_y = [10,7,10]
    For each perturbation we create a PerturbationTS.
    This store piv_x, piv_y, idx, line_version,pred_class.


    Finally, we will return a list of all the PerturbationTS.
    :return: List[PerturbationTS].
    """
    org_ts_length = len(line_version_original)
    all_perturbations = create_x_y_perturbation(org_pivots_y=pivots_y_original, org_pivots_x=pivots_x_original,
                                                ts_length=org_ts_length, epsilon=epsilon)
    # calculate line version
    for perturbation in all_perturbations:
        perturbation.set_line_version(ts_length=org_ts_length)
    return all_perturbations


def make_perturbations_and_get_class(pivots_y_original: List[float], pivots_x_original: List[int],
                                     line_version_original: List[float] | np.ndarray, epsilon: float, model_name: str):
    # Make perturbations
    all_perturbations = make_perturbations(pivots_x_original=pivots_x_original, pivots_y_original=pivots_y_original,
                                           line_version_original=line_version_original, epsilon=epsilon)
    # get class
    all_lines = [perturbation.line_version for perturbation in all_perturbations]

    all_pred_class = model_batch_classify(model_name=model_name, batch_of_timeseries=all_lines)
    for perturbation, pred_class in zip(all_perturbations, all_pred_class):
        perturbation.set_class(pred_class)

    return all_perturbations


def get_perturbations_scatter_params(all_perturbations: List[PerturbationTS]) -> List[ScatterParams]:
    all_scatter_params = {
        0: [],
        1: []
    }
    for perturbation in all_perturbations:
        if perturbation.pred_class == 0:
            all_scatter_params[0].append(perturbation)
        else:
            all_scatter_params[1].append(perturbation)

    all_0_x = [perturbation.new_x for perturbation in all_scatter_params[0]]
    all_0_y = [perturbation.new_y for perturbation in all_scatter_params[0]]

    all_1_x = [perturbation.new_x for perturbation in all_scatter_params[1]]
    all_1_y = [perturbation.new_y for perturbation in all_scatter_params[1]]

    alpha = 2e-3
    scatter_0 = ScatterParams(x_values=all_0_x, y_values=all_0_y,
                              color=to_rgba(class_to_color(0), alpha=alpha))
    scatter_1 = ScatterParams(x_values=all_1_x, y_values=all_1_y, color=to_rgba(class_to_color(1), alpha=alpha))

    scatter_params = [scatter_0, scatter_1]

    return scatter_params


def plot_perturbations_org_approx(perturbations: List[PerturbationTS], original_ts: List[float] | np.ndarray,
                                  approximation_ts: List[float], model_name: str, pivot_x_org: List[int],
                                  pivot_y_org: List[float],
                                  pivot_x_approx: List[int], pivot_y_approx: List[int]):
    scatter_params = get_perturbations_scatter_params(perturbations)

    # Get TS param for original
    ts_param_org = get_ts_param_org(y_org=original_ts, model_name=model_name)
    # Get TS Param for approx
    ts_param_approx = get_ts_param_approx(y_approx=approximation_ts, model_name=model_name)
    ts_params = [ts_param_org, ts_param_approx]

    # Add elipses on the pivot points
    ellipsis_org = make_all_ellipse_param(x_pivots=pivot_x_org, y_pivots=pivot_y_org, inner=True)
    # Add hallow elipses on the edge points
    ellipsis_approx = make_all_ellipse_param(x_pivots=pivot_x_approx, y_pivots=pivot_y_approx, inner=False)
    all_ellipse_params = ellipsis_org + ellipsis_approx

    # Make the plot and display it
    title = "Best Title there is!"
    save_file = "testFile"
    x_min = min(perturbation.new_x for perturbation in perturbations)
    x_max = max(perturbation.new_x for perturbation in perturbations)

    y_min = min(perturbation.new_y for perturbation in perturbations)
    y_max = max(perturbation.new_y for perturbation in perturbations)

    x_lim = (x_min - 1, x_max + 1)
    y_lim = (y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10)  # Some extra room

    PlotParams(ts_params=ts_params, scatter_params=scatter_params, ellipse_params=all_ellipse_params, title=title,
               save_file=save_file, display=True,
               x_lim=x_lim,
               y_lim=y_lim).make_plot()


def test(instance_nr, model_name, dataset_name):
    pivot_points_x, pivot_points_y, ts, min_y, max_y = get_time_series_and_seg_pivots(dataset_name=dataset_name,
                                                                                      instance_nr=instance_nr)
    pivot_point_x_org = pivot_points_x.copy()
    pivot_point_y_org = pivot_points_y.copy()
    # best_fit_ys_org = pivot_points_y.copy()
    # best_fit_points_org = pivot_point_x.copy()
    approx_line = interpolate_points_to_line(len(ts), pivot_points_x, pivot_points_y)
    pivot_points_x[0] = 0
    pivot_points_x[-1] = len(ts) - 1

    pivot_points_y[0] = approx_line[0]
    pivot_points_y[-1] = approx_line[-1]

    epsilon_div = (max_y - min_y) / 5
    all_perturbations = make_perturbations_and_get_class(pivots_x_original=pivot_points_x,
                                                         pivots_y_original=pivot_points_y,
                                                         line_version_original=ts, model_name=model_name,
                                                         epsilon=epsilon_div)
    plot_perturbations_org_approx(perturbations=all_perturbations, original_ts=ts, approximation_ts=approx_line,
                                  model_name=model_name, pivot_x_org=pivot_point_x_org, pivot_y_org=pivot_point_y_org,
                                  pivot_x_approx=pivot_points_x, pivot_y_approx=pivot_points_y)
    # perturbation_by_x = _create_perturbation_all_x_all_y_dict(pivots_x=best_fit_points, pivots_ys=best_fit_ys,
    #                                                          epsilon=epsilon_div)


#
# line_version_dict = {}
# for x in perturbation_by_x.keys():
#
#    line_version_dict[x + 1] = {}
#    if x not in line_version_dict:
#        line_version_dict[x] = {}
#    if x - 1 not in line_version_dict:
#        line_version_dict[x - 1] = {}
#    for y in perturbation_by_x[x].keys():
#        line_version_xy = interpolate_points_to_line(ts_length=len(ts), x_selected=best_fit_points,
#                                                     y_selected=perturbation_by_x[x][y])
#        line_version_dict[x][y] = line_version_xy
#
#        if x != 0 and (x - 1) not in best_fit_points:
#            new_x_pivots = [idx if x != idx else idx - 1 for idx in best_fit_points]
#            line_before_version_xy = interpolate_points_to_line(ts_length=len(ts), x_selected=new_x_pivots,
#                                                                y_selected=perturbation_by_x[x][y])
#
#            line_version_dict[x - 1][y] = line_before_version_xy
#
#        if x != best_fit_points[-1] and (x + 1) not in best_fit_points:
#            new_x_pivots = [idx if x != idx else idx + 1 for idx in best_fit_points]
#            line_before_version_xy = interpolate_points_to_line(ts_length=len(ts), x_selected=new_x_pivots,
#                                                                y_selected=perturbation_by_x[x][y])
#
#            line_version_dict[x + 1][y] = line_before_version_xy
#        # DO the same other side!
#
# x_y_line = []
# for x_idx in line_version_dict.keys():
#    for y_value in line_version_dict[x_idx].keys():
#        curr_line = line_version_dict[x_idx][y_value]
#        # class_curr = model_classify(model_name="Chinatown_1000.keras", time_series=curr_line)
#        # dict_class[x_idx][y_value] = class_curr
#        x_y_line.append((x_idx, y_value, curr_line))
#
# dict_class = {}
# just_lines_values = [line for x, y, line in x_y_line]
# class_all = model_batch_classify(model_name=model_name, batch_of_timeseries=just_lines_values)
# x_y_c = [(x, y, c) for (x, y, line), c in zip(x_y_line, class_all)]
# class_org = model_classify(model_name=model_name, time_series=ts)
# class_approx = model_classify(model_name=model_name, time_series=approx_line)
#
## x_to_ymin_ymax = find_continues_x_length(x_y_c=x_y_c, target_class=class_approx)
#
## x_to_y = [(x, x_to_ymin_ymax[x][0], x_to_ymin_ymax[x][1]) for x in x_to_ymin_ymax.keys()]
## vert_x = [x for x, min_y, max_y in x_to_y]
## vert_y_min = [min_y for x, min_y, max_y in x_to_y]
## vert_y_max = [max_y for x, min_y, max_y in x_to_y]
#
# fig, ax = plt.subplots()
#
## plt.vlines(x=vert_x, ymin=vert_y_min, ymax=vert_y_max, colors=class_to_color(class_approx))
# X_0 = [x for x, y, c in x_y_c if c == 0]
# Y_0 = [y for x, y, c in x_y_c if c == 0]
# X_1 = [x for x, y, c in x_y_c if c == 1]
# Y_1 = [y for x, y, c in x_y_c if c == 1]
## print(X_1, Y_1)
# plt.clf()
# plt.scatter(X_0, Y_0, color=class_to_color(0))
# plt.scatter(X_1, Y_1, color=class_to_color(1))
# org_color = class_to_color(class_org)
# plt.plot(list(range(len(ts))), ts, color=org_color)
# approx_color = class_to_color(class_approx)
# plt.plot(list(range(len(approx_line))), approx_line, "--", color=approx_color)
# line_above = approx_line + epsilon_div
# line_below = approx_line - epsilon_div
#
# plt.title(f"{dataset_name}, Feature Attribution instance nr: {instance_nr}")
#
## Make it look nice
# plt.plot(list(range(len(line_above))), line_above, color='black')
# plt.plot(list(range(len(line_below))), line_below, color='black')
# y_lim_max = max_y + epsilon_div
# y_lim_min = min_y - epsilon_div
# plt.ylim((y_lim_min, y_lim_max))
# plt.xlim((-1, 24))
# for i, x in enumerate(best_fit_points_org):
#    y = best_fit_ys_org[i]
#    x_axsis_r = 0.4
#    y_axsis_r = x_axsis_r * (y_lim_max - y_lim_min) / (24 - (-1))
#    ellipse = patches.Ellipse((x, y), x_axsis_r, y_axsis_r, fill=True, color='grey')
#    ax.add_patch(ellipse)
#
# for i, x in enumerate(best_fit_points):
#    y = best_fit_ys[i]
#    x_axsis_r = 0.6
#    y_axsis_r = x_axsis_r * (y_lim_max - y_lim_min) / (24 - (-1))
#    ellipse = patches.Ellipse((x, y), x_axsis_r, y_axsis_r, fill=False, color='black')
#    ax.add_patch(ellipse)
# plt.savefig(f"PyPlots/{dataset_name}/{instance_nr}")
# plt.show()
#
#
def run():
    model_name = "Chinatown_1000.keras"
    dataset_name = "Chinatown"
    os.makedirs(f"PyPlots/{dataset_name}", exist_ok=True)
    for i in range(0, 100, 1):
        test(instance_nr=i, model_name=model_name, dataset_name=dataset_name)


if __name__ == '__main__':
    run()
