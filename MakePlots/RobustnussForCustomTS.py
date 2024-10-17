import numpy as np

from ProtoTypes.ProtoTypes import get_prototypes

from utils.data import get_time_series, get_min_max_from_dataset_name
from utils.folder import make_folder

from selectBestApproximation.critieria import get_best_approximation

from localRobustness.LocalRobustNess import get_local_robust_score

from Perturbations.dataTypes import SegmentedTS

from models.loadModel import model_classify


def run():
    dataset = "ECG200"
    model_name = "ECG200_100.keras"
    instance_nr = 2
    class_0, class_1 = get_prototypes(dataset_name=dataset, model_name=model_name)
    # time_series = get_time_series(dataset_name=dataset, instance_nr=2)
    if dataset != "Chinatown":
        instance_nr = class_0[0]
    robustness_resolution = 1000
    k_count = 1
    best_approximation = get_best_approximation(dataset_name=dataset, model_name=model_name, instance_nr=instance_nr,
                                                alpha=0.01, beta=0.001, gamma=100,
                                                robustness_resolution=robustness_resolution,
                                                k=k_count)
    robustness_resolution = 10000

    max_value = 250
    y_value_0 = max_value
    y_value_5 = 0
    new_y_pivots = [y_value_0, y_value_5]
    new_x_pivots = [0, 5]
    for x, y in zip(best_approximation.x_pivots, best_approximation.y_pivots):
        if x <= 5:
            continue
        else:
            new_y_pivots.append(y)
            new_x_pivots.append(x)

    new_time_series_seg = SegmentedTS(
        x_pivots=new_x_pivots,
        y_pivots=new_y_pivots,
        ts_length=len(best_approximation.line_version)
    )
    y_min, y_max = get_min_max_from_dataset_name(dataset_name=dataset)
    epsilon = (y_max - y_min) / 5  # (y_max - y_min) / 10
    folder_name = "RobustnessPlot"
    make_folder(f"PyPlots/{dataset}/{folder_name}")
    curr_class = model_classify(time_series=best_approximation.line_version, model_name=model_name)
    get_local_robust_score(approximation=best_approximation,
                           model_name=model_name,
                           target_class=curr_class,
                           lim_y=(y_min - epsilon, y_max),
                           epsilon=epsilon,
                           save_file=f"{dataset}/{folder_name}/{instance_nr}_{max_value}_{robustness_resolution}_wrong",
                           verbose=True,
                           title="",  # f"{dataset} ts[0]={max_value}",
                           k=robustness_resolution
                           )


if __name__ == '__main__':
    run()
