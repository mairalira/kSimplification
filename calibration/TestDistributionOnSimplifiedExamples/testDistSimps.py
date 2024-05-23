from simplify.DPcustomAlgoKSmallest import solve_and_find_points
import numpy as np
from generate_approximation_for_all import interpolate_points_to_line, get_min_and_max, calculate_seg_punishment_c
from dataSet.load_data import load_dataset_ts
from calibration.calibration import show_calibration
from models.loadModel import load_keras_model, model_classify


def test_dist_simp(orgTimeSeries: np.array, c: int, k: int, saveImg=False):
    X = list(range(len(orgTimeSeries)))
    all_selected_points, all_ys = solve_and_find_points(X=X, Y=orgTimeSeries, c=c, K=k, saveImg=saveImg)

    all_interpolations = []
    for i, (selected_points, ys) in enumerate(zip(all_selected_points, all_ys)):
        inter_ts = interpolate_points_to_line(ts_lenght=len(orgTimeSeries), x_selcted=selected_points, y_selcted=ys)
        all_interpolations.append(inter_ts)

    return np.array(all_interpolations)


def run():
    all_time_series = load_dataset_ts("Chinatown", data_type="TEST")

    c_percentage = 200
    my_c = calculate_seg_punishment_c(all_time_series, c_percentage)  # Chinatown: 1

    my_k = 1000000

    ts_nr = 50
    ts = all_time_series[ts_nr]
    all_interpolations = test_dist_simp(ts, c=my_c, k=my_k)

    model_name = "Chinatown.keras"
    model = load_keras_model(model_name)
    import random
    y_test = np.array([random.randint(0, 1) for _ in range(len(all_interpolations))])
    show_calibration(model=model, x_test=all_interpolations, y_test=y_test,
                     model_name=model_name)
    print(model.predict(np.array([ts])))


if __name__ == '__main__':
    run()
