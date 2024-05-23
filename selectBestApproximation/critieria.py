from models.loadModel import model_confidence as get_confidence
import numpy as np


def get_similarity(ts1, ts2):
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    dist = np.linalg.norm(ts1 - ts2)
    return dist


def score_approximation(approximation, original, model_name):
    """
    Score an input approximation
    :param approximation:
    :param original:
    :param model_name:
    :return score:
    """
    similarity_factor = 0.5
    similarity = get_similarity(approximation, original)

    confidence = get_confidence(approximation, model_name)

    return similarity * similarity_factor + (1 - similarity_factor) * confidence
