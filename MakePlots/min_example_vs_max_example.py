from typing import List

import numpy as np

from simplify.DPcustomAlgoKSmallest import solve_and_find_points

from utils.data import get_min_max_from_dataset_name, dataset_sensitive_c, set_c_manually
from utils.data import load_dataset_ts
from utils.scoring_functions import score_closeness, score_simplicity
from selectBestApproximation.critieria import score_approximation
from models.loadModel import model_classify

from Perturbations.dataTypes import SegmentedTS

import matplotlib.pyplot as plt


def score_closeness_and_simplicity(approximation: SegmentedTS, original: List[float], alpha: float, beta: float,
                                   distance_weight: float):
    closeness_score = score_closeness(ts1=approximation.line_version, ts2=original, distance_weight=distance_weight,
                                      alpha=alpha)
    simplicity = score_simplicity(approximation=approximation, beta=beta)
    return closeness_score + simplicity


def min_max_score(single_ts: List[float] | np.ndarray, dataset_name: str, model_name: str, alpha: float, k: int,
                  gamma=1, beta=0.005):
    set_c_manually(beta)
    min_y, max_y = get_min_max_from_dataset_name(dataset_name=dataset_name)
    distance_weight = abs(max_y - min_y)
    c = dataset_sensitive_c(dataset=dataset_name, distance_weight=distance_weight)
    # print(alpha, c, gamma)
    all_selected_points, all_ys = solve_and_find_points(X=list(range(len(single_ts))), Y=single_ts, c=c, K=k,
                                                        distance_weight=distance_weight,
                                                        alpha=alpha)
    to_segs = [SegmentedTS(x_pivots=x_pivots, y_pivots=y_pivots, ts_length=len(single_ts)) for (x_pivots, y_pivots) in
               zip(all_selected_points, all_ys)]

    min_approximation = to_segs[0]
    closeness_score_min = score_closeness(ts1=min_approximation.line_version, ts2=single_ts,
                                          alpha=alpha,
                                          distance_weight=distance_weight)
    simplicity_min = score_simplicity(approximation=min_approximation, beta=c)

    plt.plot(min_approximation.x_pivots, min_approximation.y_pivots, "--")

    # print(min_approximation.y_pivots)

    max_approximation = to_segs[-1]
    # print(max_approximation.y_pivots)
    closeness_score_max = score_closeness(ts1=max_approximation.line_version, ts2=single_ts,
                                          alpha=alpha,
                                          distance_weight=distance_weight)
    simplicity_max = score_simplicity(approximation=max_approximation, beta=c)
    plt.plot(max_approximation.x_pivots, max_approximation.y_pivots, "-")

    min_score = closeness_score_min + simplicity_min
    max_score = closeness_score_max + simplicity_max
    print(min_score, closeness_score_min, simplicity_min)
    print(max_score, closeness_score_max, simplicity_max)

    with open(f"PyPlots/{dataset_name}/DP/results_{k}.csv", "w") as f:
        f.write(
            f"{str(min_score)},{closeness_score_min},{simplicity_min}\n{str(max_score)},{closeness_score_max},{simplicity_max}")


def ecg():
    dataset_name = "ECG200"
    model_name = "ECG200_100.keras"
    alpha = 0.01
    beta = 0.0005
    gamma = 0

    all_ts = load_dataset_ts(dataset_name=dataset_name, data_type="TEST")
    nr = 55
    single_ts = all_ts[nr]

    for i in range(1, 8):
        min_max_score(single_ts=single_ts, dataset_name=dataset_name, model_name=model_name, alpha=alpha, gamma=gamma,
                      beta=beta, k=10 ** i)
        print(i)


if __name__ == '__main__':
    ecg()
