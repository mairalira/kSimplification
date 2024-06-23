import numpy as np
from numpy.random import PCG64
from typing import List, Tuple

from utils.data import load_dataset_ts, get_min_and_max

from visualization.plotting import PlotParams, TSParam

from models.loadModel import model_batch_classify


def store_test_instances(test_instances, indexes, dataset: str, model_name: str):
    all_preds = model_batch_classify(model_name=model_name, batch_of_timeseries=test_instances)
    with open(f"TestInstance/{dataset}.csv", 'w') as file:
        for test_instance, idx, pred_clas in zip(test_instances, indexes, all_preds):
            test_instance_str = ":".join([str(val) for val in test_instance])

            full_line = test_instance_str + "," + str(idx) + "," + str(pred_clas)
            file.write(full_line + "\n")


def get_k_random_instances(k: int, ts_list: List[List[float]] | np.ndarray):
    rng = np.random.Generator(PCG64(41))  # ORG: np.random.Generator(PCG64(42))

    indexes = range(len(ts_list))
    instances = rng.choice(indexes, size=k, replace=False)

    return ts_list[instances], instances


def plot_test_instances(ts_list: List[List[float]] | np.ndarray, idxes: List[int] | np.ndarray, dataset: str,
                        y_lim: Tuple[float, float]):
    ts_len = len(ts_list[0])
    x_values = [i for i in range(ts_len)]
    for ts, idx in zip(ts_list, idxes):  # type: List[float], int
        ts_param = TSParam(x_values=x_values, y_values=ts, fmat="-", color="black")
        PlotParams(ts_params=[ts_param],
                   save_file=f"{dataset}/Test/{idx}",
                   y_lim=y_lim).make_plot()


def make_test_instances(dataset: str, model_name: str, k: int = 10, ):
    ts_list = load_dataset_ts(dataset_name=dataset, data_type="TEST")
    min_y, max_y = get_min_and_max(ts_all=ts_list)
    test_instances, indexes = get_k_random_instances(ts_list=ts_list, k=k)
    plot_test_instances(ts_list=test_instances, idxes=indexes, dataset=dataset, y_lim=(min_y, max_y))
    store_test_instances(test_instances=test_instances, indexes=indexes, dataset=dataset, model_name=model_name)


def run():
    datasets = ["Chinatown", "ECG200", "ItalyPowerDemand"]
    for dataset in datasets:
        model_name = dataset + "_100.keras"
        make_test_instances(dataset=dataset, k=10, model_name=model_name)


if __name__ == "__main__":
    run()
