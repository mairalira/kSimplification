def get_confidence_of_same_class_ts(target_class: int, all_class: [int], all_confidence: [float], all_ts: [float]) -> \
        [([float], float)]:
    """
    :param target_class:
    :param all_class:
    :param all_confidence:
    :param all_ts:
    :return: [(TS, confidence)]
    """
    ts_confidence = []
    for curr_class, curr_confidence, curr_ts in zip(all_class, all_confidence, all_ts):
        if curr_class == target_class:
            ts_confidence.append((curr_ts, curr_confidence))
    return ts_confidence


def class_to_color(predicted_class: int) -> str:
    class_to_color_dict = {
        1: "red",
        0: "blue"
    }
    return class_to_color_dict[predicted_class]
