from models.loadModel import model_confidence as get_confidence
from localRobustness.LocalRobustNess import get_local_robust_score_approx
import numpy as np
from dataSet.load_data import load_dataset_ts
from simplify.DPcustomAlgoKSmallest import solve_and_find_points
from utils.data import dataset_sensitive_c
from models.loadModel import model_classify
from utils.line import convert_all_points_to_lines
from tqdm import tqdm
import numpy as np
from utils.model import class_to_color


def sigmoid(x, k=0.0001):
    return 1 / (1 + np.exp(-k * x))


def get_similarity(ts1, ts2):
    # Euclidean distance
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    dist = np.linalg.norm(ts1 - ts2)

    return dist


def get_similarity_score(ts1, ts2, minDist, maxDist):
    similarity = (get_similarity(ts1, ts2) - minDist) / (maxDist - minDist)
    return similarity


def score_approximation(approximation, original, model_name, target_class: int, alpha: float, min_dist: float,
                        max_dist: float):
    """
    Score an input approximation
    :param alpha: currBest
    :param target_class:
    :param approximation:
    :param original:
    :param model_name:
    :return score:
    """
    similarity_factor = 0.5
    similarity = get_similarity_score(approximation, original, minDist=min_dist, maxDist=max_dist)
    if similarity * similarity_factor > alpha:
        return float("inf"), None, None
    robustness_score = get_local_robust_score_approx(approximation, model_name, k=10 ** 4, target_class=target_class)
    robustness_score = 1 - robustness_score  # Robustness of 1 means every single perturbation had the target class
    score = similarity * similarity_factor + (1 - similarity_factor) * robustness_score
    return score, similarity, robustness_score


def test():
    dataset_name = "Chinatown"
    model_name = "Chinatown_1000.keras"
    all_ts = load_dataset_ts(dataset_name, data_type="TEST")
    single_ts = all_ts[0]
    target_class = model_classify(model_name=model_name, time_series=single_ts)
    c = dataset_sensitive_c(ts_all=all_ts, percentage_c=200)
    nr_of_approximation = 10 ** 4

    all_selected_points, all_ys = solve_and_find_points(X=list(range(len(single_ts))), Y=single_ts, c=c,
                                                        K=nr_of_approximation)
    all_line_versions = convert_all_points_to_lines(ts_length=len(single_ts), all_x_selected=all_selected_points,
                                                    all_y_selected=all_ys)
    all_dist = [get_similarity(ts1=single_ts, ts2=approx) for approx in all_line_versions]
    min_dist = min(all_dist)
    max_dist = max(all_dist)

    import matplotlib.pyplot as plt

    best_approx = None
    best_approx_score = float("inf")
    for line_approx in tqdm(all_line_versions):
        score, similarity, robustness_score = score_approximation(approximation=line_approx, original=single_ts,
                                                                  model_name=model_name,
                                                                  target_class=target_class, alpha=best_approx_score,
                                                                  min_dist=min_dist,
                                                                  max_dist=max_dist)
        if score < best_approx_score:
            best_approx = line_approx
            best_approx_score = score
            print(
                f"Best_score {best_approx_score}, best_approx {best_approx}\nSimilarity: {similarity} robustness: {robustness_score}")
            plt.clf()
            plt.plot(list(range(len(best_approx))), best_approx, color=class_to_color(target_class))
            plt.plot(list(range(len(single_ts))), single_ts, "--", color=class_to_color(target_class))
            plt.show()


if __name__ == '__main__':
    test()
