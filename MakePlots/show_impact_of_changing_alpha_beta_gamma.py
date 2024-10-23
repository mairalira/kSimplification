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
                                         model_name: str, verbose=False, robustness_resolution=10 ** 4, k=1 * 10 ** 4):
    best_approximation = get_best_approximation(dataset_name=dataset_name, model_name=model_name,
                                                instance_nr=instance_nr,
                                                alpha=alpha, beta=beta, gamma=gamma, early_stop=(gamma >= 0),
                                                k=k, verbose=verbose,
                                                robustness_resolution=robustness_resolution)
    print(instance_nr, )
    min_y, max_y = get_min_max_from_dataset_name(dataset_name=dataset_name)
    ts_param = get_ts_param_approx(best_approximation.line_version, model_name=model_name)
    org_ts = get_time_series(dataset_name=dataset_name, instance_nr=instance_nr)
    ts_org_param = get_ts_param_org(y_org=org_ts, model_name=model_name)
    folder_name = "AlphaBetaGamma"
    make_folder(f"PyPlots/{dataset_name}/{folder_name}")
    title = f"Nr:{instance_nr}, Alpha:{alpha}, Beta:{beta}, Gamma:{gamma}"
    PlotParams(
        ts_params=[ts_param, ts_org_param],
        display=True,
        title=title,
        y_lim=(min_y, max_y),
        save_file=f"{dataset_name}/{folder_name}/nr_{instance_nr}_a_{alpha}_b_{beta}_g_{gamma}_k_{k}"
    ).make_plot()


def china_town_config(instance_nr: int, dataset_name: str, model_name: str):
    robustness_resolution = 2 * 10 ** 3
    k = 10 ** 4
    verbose = False
    alpha_mid = 0.001
    beta_mid = 10 ** -5
    gamma_mid = 0.01

    # Config 1 mid alpha, low beta, mid gamma
    alpha = alpha_mid
    beta = round(beta_mid / 10, 8)
    gamma = gamma_mid
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution,
                                         k=k)

    # Config 2 mid alpha, mid beta, mid gamma
    alpha = alpha_mid
    beta = beta_mid
    gamma = gamma_mid
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution,
                                         k=k)

    # Config 3 mid alpha, mid beta, high gamma
    alpha = alpha_mid
    beta = beta_mid
    gamma = round(gamma_mid * 100000, 8)
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution,
                                         k=k)
    # Config 4 mid alpha, high beta, mid gamma
    alpha = alpha_mid
    beta = round(beta_mid * 10, 8)
    gamma = gamma_mid
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution,
                                         k=k)


def ECG200_config(instance_nr: int, dataset_name: str, model_name: str):
    robustness_resolution = 10 ** 2
    k = 2 * 10 ** 4
    verbose = False
    alpha_mid = 0.01
    beta_mid = 0.0001  # 0.0001
    gamma_mid = 0.001

    # Testing
    alpha_mid = 1
    beta_mid = 0
    gamma_mid = 0

    # Config 1 mid alpha, low beta, mid gamma
    # alpha = alpha_mid
    # beta = round(beta_mid / 10, 8)
    # gamma = gamma_mid
    # make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
    #                                      dataset_name=dataset_name, model_name=model_name, verbose=verbose,
    #                                      robustness_resolution=robustness_resolution,
    #                                      k=k)

    ## Config 2 mid alpha, mid beta, mid gamma
    alpha = alpha_mid
    beta = beta_mid
    gamma = gamma_mid
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution,
                                         k=k)
    ## Config 3 mid alpha, mid beta, high gamma
    alpha = alpha_mid
    beta = beta_mid
    gamma = round(gamma_mid * 100, 8)
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution,
                                         k=k)

    # Config 4 mid alpha, high beta, mid gamma
    # alpha = alpha_mid
    # beta = round(beta_mid * 10, 8)
    # gamma = gamma_mid
    # make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
    #                                      dataset_name=dataset_name, model_name=model_name, verbose=verbose,
    #                                      robustness_resolution=robustness_resolution,
    #                                      k=k)


def IDP_config(instance_nr: int, dataset_name: str, model_name: str):
    robustness_resolution = 1 * 10 ** 3
    k = 5 * 10 ** 3
    verbose = False
    alpha_mid = 0.001
    beta_mid = 0.000075
    gamma_mid = 0.00001

    # Config 2 mid alpha, mid beta, mid gamma
    alpha = alpha_mid
    beta = beta_mid
    gamma = gamma_mid
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution, k=k)

    ## Config 0 mid alpha, mid beta, high gamma
    alpha = alpha_mid
    beta = beta_mid
    gamma = round(gamma_mid * 10 ** 5, 8)
    make_plot_by_chosen_alpha_beta_gamma(alpha=alpha, beta=beta, gamma=gamma, instance_nr=instance_nr,
                                         dataset_name=dataset_name, model_name=model_name, verbose=verbose,
                                         robustness_resolution=robustness_resolution, k=k)


def different_configurations(instance_nr: int, dataset_name: str, model_name: str):
    if dataset_name == "Chinatown":
        china_town_config(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)
    if dataset_name == "ItalyPowerDemand":
        IDP_config(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)
    if dataset_name == "ECG200":
        ECG200_config(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)


def run_with_multiple_instance_numbers(dataset_name, model_name):
    selected_idx_class_0, selected_idx_class_1 = get_prototypes(dataset_name=dataset_name, model_name=model_name)

    all_idxes = selected_idx_class_0 + selected_idx_class_0
    if dataset_name == "Chinatown":
        all_idxes = [2]
    elif dataset_name == "ItalyPowerDemand":
        all_idxes = [290]
    elif dataset_name == "ECG200":
        all_idxes = selected_idx_class_0 + selected_idx_class_1
        rng = np.random.Generator(PCG64(42))
        instance_nr = rng.choice(all_idxes)
        all_idxes = [instance_nr]
    for instance_nr in all_idxes:
        different_configurations(instance_nr=instance_nr, dataset_name=dataset_name, model_name=model_name)

    # if dataset_name == "Chinatown":
    #    instance_nr = 2  # Prototype and one we know we have good results for!
    # elif dataset_name == "ItalyPowerDemand":
    #    all_idxes = selected_idx_class_1
    #    rng = np.random.Generator(PCG64(42))
    #    instance_nr = rng.choice(all_idxes)
    # else:
    #    all_idxes = selected_idx_class_1  # + selected_idx_class_0
    #    rng = np.random.Generator(PCG64(42))
    #    instance_nr = rng.choice(all_idxes)


def multiple_datasets():
    datasets = ["Chinatown", "ItalyPowerDemand", "ECG200"]
    models = ["Chinatown_100.keras", "ItalyPowerDemand_100.keras", "ECG200_100.keras"]
    for dataset_name, model in zip(datasets, models):
        run_with_multiple_instance_numbers(dataset_name=dataset_name, model_name=model)


if __name__ == '__main__':
    multiple_datasets()
