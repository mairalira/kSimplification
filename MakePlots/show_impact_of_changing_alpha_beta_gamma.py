import sys
import os

# Add the directory containing 'Perturbations' to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
parent_dir = os.path.dirname(script_dir)  # One level up from the script directory
sys.path.append(parent_dir)

import numpy as np
from numpy.random import PCG64

from Perturbations.dataTypes import SegmentedTS

from selectBestApproximation.critieria import get_best_approximation

from visualization.plotting import TSParam, PlotParams
from visualization.getTSParam import get_ts_param_approx, get_ts_param_org

from utils.data import get_min_max_from_dataset_name, get_time_series
from utils.folder import make_folder

from ProtoTypes.ProtoTypes import get_prototypes


def make_plot_by_chosen_alpha_beta_gamma(alpha: float, beta: float, gamma: float, instance_nr: int, dataset_name: str,
                                         model_name: str):
    k = 1 * 10 ** 6
    best_approximation = get_best_approximation(dataset_name=dataset_name, model_name=model_name,
                                                instance_nr=instance_nr,
                                                alpha=alpha, beta=beta, gamma=gamma, early_stop=(gamma >= 0), k=k)
    min_y, max_y = get_min_max_from_dataset_name(dataset_name=dataset_name)
    ts_param = get_ts_param_approx(best_approximation.line_version, model_name=model_name)
    org_ts = get_time_series(dataset_name=dataset_name, instance_nr=instance_nr)
    ts_org_param = get_ts_param_org(y_org=org_ts, model_name=model_name)
    folder_name = "AlphaBetaGamma"
    make_folder(f"PyPlots/{dataset_name}/{folder_name}")
    PlotParams(
        ts_params=[ts_param, ts_org_param],
        display=True,
        title=f"Nr:{instance_nr}, Alpha:{alpha}, Beta:{beta}, Gamma:{gamma}",
        y_lim=(min_y, max_y),
        save_file=f"{dataset_name}/{folder_name}/a_{alpha}_b_{beta}_g_{gamma}_k_{k}"
    ).make_plot()


def custom_config(instance_nr: int, dataset_name: str, model_name: str):
    # Config -1
    alpha = 0.1
    beta = 0.001
    gamma = -10
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 0
    alpha = 0.1
    beta = 0.001
    gamma = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    ## Config 1 Mid $\alpha$ - Low $\beta$ - Mid $\gamma$
    # alpha = 0.1
    # beta = 0.0001
    # gamma = 0.01
    # make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
    #                                     dataset_name=dataset_name, model_name=model_name)
    #
    ## Config 2
    # alpha = 0.1
    # beta = 0.005
    # gamma = 0.01
    # make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
    #                                     dataset_name=dataset_name, model_name=model_name)
    ## Config 3
    # alpha = 0.1
    # beta = 0.001
    # gamma = 0.01
    # make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
    #                                     dataset_name=dataset_name, model_name=model_name)
    #
    ## Config 4
    # alpha = 0.1
    # beta = 0.001
    # gamma = 0.1
    # make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
    #                                     dataset_name=dataset_name, model_name=model_name)
    #
    ## Config 5
    # alpha = 0.1
    # beta = 0.001
    # gamma = 10
    # make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
    #                                     dataset_name=dataset_name, model_name=model_name)


def different_alpha_configurations(instance_nr: int, dataset_name: str, model_name: str):
    # Config 1
    alpha = 0.01
    beta = 0.01
    gamma = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 2
    alpha = 0.1
    beta = 0.01
    gamma = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 3
    alpha = 1
    beta = 0.01
    gamma = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 4
    alpha = 10
    beta = 0.01
    gamma = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 5
    alpha = 100
    beta = 0.01
    gamma = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)


def different_beta_configurations(instance_nr: int, dataset_name: str, model_name: str):
    alpha = 0.1
    gamma = 0.01

    # Config 1
    beta = 0.0001
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 2
    beta = 0.001
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 3
    beta = 0.005
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 4
    beta = 0.01
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 5
    beta = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)


def different_gamma_configurations(instance_nr: int, dataset_name: str, model_name: str):
    alpha = 0.05
    beta = 0.001

    # Config 1
    gamma = 0.001
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 2
    gamma = 0.01
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)
    # Config 3
    gamma = 0.1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 4
    gamma = 1
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 5
    gamma = 10
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)
    # Config 6
    gamma = 100
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)

    # Config 6
    gamma = 1000
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name)


def different_configurations(instance_nr: int, dataset_name: str, model_name: str):
    # different_alpha_configurations(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)
    # different_beta_configurations(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)
    # different_gamma_configurations(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)
    custom_config(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)


def run_with_multiple_instance_numbers(dataset_name, model_name):
    selected_idx_class_0, selected_idx_class_1 = get_prototypes(dataset_name=dataset_name, model_name=model_name)

    all_idxes = selected_idx_class_0  # + selected_idx_class_0

    if dataset_name == "Chinatown":
        instance_nr = 2  # Prototype and one we know we have good results for!
    elif dataset_name == "ItalyPowerDemand":
        all_idxes = selected_idx_class_0
        rng = np.random.Generator(PCG64(42))
        instance_nr = rng.choice(all_idxes)
    else:
        all_idxes = selected_idx_class_1  # + selected_idx_class_0
        rng = np.random.Generator(PCG64(42))
        instance_nr = rng.choice(all_idxes)

    different_configurations(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)


def multiple_datasets():
    datasets = ["Chinatown", "ItalyPowerDemand", "ECG200"]
    models = ["Chinatown_100.keras", "ItalyPowerDemand_100.keras", "ECG200_100.keras"]
    for dataset_name, model in zip(datasets, models):
        run_with_multiple_instance_numbers(dataset_name=dataset_name, model_name=model)


if __name__ == '__main__':
    multiple_datasets()
