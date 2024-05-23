import tensorflow as tf
from train_models.FCN_model import Classifier_FCN
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, f1_score
import ml_insights as mli
import matplotlib.pyplot as plt
from models.loadModel import load_keras_model
from dataSet.load_data import load_dataset_ts, load_dataset_labels


def show_calibration(model, x_test, y_test, model_name: str):
    preds_uncali = np.array([y[1] for y in model.predict(x_test)])
    print(f"LightGBM logloss on the test set: {log_loss(y_test, preds_uncali):.5f}")
    print(f"LightGBM ROC-AUC on the test set: {roc_auc_score(y_test, preds_uncali):.5f}")
    print(f"LightGBM F1 on the test set: {f1_score(y_test, [np.argmax(y) for y in model.predict(x_test)]):.5f}")
    plt.figure(figsize=(15, 5))
    rd = mli.plot_reliability_diagram(y_test, preds_uncali, show_histogram=True)
    # plt.title(f"{model_name}")
    plt.show()
    plt.savefig(f"calibration/imgs/{model_name}.png")
    # return predictions


if __name__ == "__main__":
    model_name = "Chinatown_1000.keras"
    model = load_keras_model(model_name)

    dataset_name = "Chinatown"
    x_test_dataset = load_dataset_ts(dataset_name, data_type="TEST")
    y_test_dataset = load_dataset_labels(dataset_name, data_type="TEST")
    print(len(y_test_dataset))
    show_calibration(model=model, x_test=x_test_dataset, y_test=y_test_dataset, model_name=model_name)
