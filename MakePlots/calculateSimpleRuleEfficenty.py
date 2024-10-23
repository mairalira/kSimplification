from typing import List

import matplotlib.pyplot as plt

from sklearn import metrics

import numpy as np

from models.loadModel import model_batch_classify
from dataSet.load_data import load_dataset_ts
from dataSet.load_data import load_dataset_labels

from utils.folder import make_folder


def ecg_rule():
    dataset_name = "ECG200"
    model_name = "ECG200_100.keras"
    training_data = load_dataset_ts(dataset_name=dataset_name, data_type="TRAIN")
    test_data = load_dataset_ts(dataset_name=dataset_name, data_type="TEST")
    validation_data = load_dataset_ts(dataset_name=dataset_name, data_type="VALIDATION")

    all_data_combined = np.concatenate((training_data, test_data, validation_data))
    all_classes_pred = model_batch_classify(batch_of_timeseries=all_data_combined, model_name=model_name)

    training_labels = load_dataset_labels(dataset_name=dataset_name, data_type="TRAIN")
    test_labels = load_dataset_labels(dataset_name=dataset_name, data_type="TEST")
    validation_labels = load_dataset_labels(dataset_name=dataset_name, data_type="VALIDATION")
    all_classes_gt = np.concatenate((training_labels, test_labels, validation_labels))
    basic_rule = []
    rule_threshold = 0.3

    for ts in all_data_combined:

        rapid_invert_change = False
        for i in range(3, 10):
            curr = np.mean(ts[i] + ts[i + 1])
            next = np.mean(ts[i + 2] + ts[i + 3])
            if next - curr > rule_threshold:
                rapid_invert_change = True
                break

        if rapid_invert_change:
            simple_rule_guess = 1
        else:
            simple_rule_guess = 0
        basic_rule.append(simple_rule_guess)

    rule_pred_CM = metrics.confusion_matrix(basic_rule, all_classes_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=rule_pred_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.ylabel(f"Basic rule >= {rule_threshold}")
    plt.xlabel("Model Pred")
    folder_name = "BasicRule"
    make_folder(f"PyPlots/{dataset_name}/{folder_name}")

    plt.savefig(f"PyPlots/{dataset_name}/{folder_name}/rule_{rule_threshold}_pred.png")
    plt.show()

    plt.clf()
    rule_gt_CM = metrics.confusion_matrix(basic_rule, all_classes_gt)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=rule_gt_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.ylabel(f"Basic rule >= {rule_threshold}")
    plt.xlabel("Ground Truth")
    plt.savefig(f"PyPlots/{dataset_name}/{folder_name}/rule_{rule_threshold}_gt.png")

    plt.show()

    plt.clf()
    pred_gt_CM = metrics.confusion_matrix(all_classes_pred, all_classes_gt)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=pred_gt_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.ylabel("Model Pred")
    plt.xlabel("Ground Truth")
    plt.savefig(f"PyPlots/{dataset_name}/{folder_name}/pred_gt.png")
    plt.show()


def IDP_rule():
    dataset_name = "ItalyPowerDemand"
    model_name = "ItalyPowerDemand_100.keras"
    training_data = load_dataset_ts(dataset_name=dataset_name, data_type="TRAIN")
    test_data = load_dataset_ts(dataset_name=dataset_name, data_type="TEST")
    validation_data = load_dataset_ts(dataset_name=dataset_name, data_type="VALIDATION")

    all_data_combined = np.concatenate((training_data, test_data, validation_data))
    all_classes_pred = model_batch_classify(batch_of_timeseries=all_data_combined, model_name=model_name)

    training_labels = load_dataset_labels(dataset_name=dataset_name, data_type="TRAIN")
    test_labels = load_dataset_labels(dataset_name=dataset_name, data_type="TEST")
    validation_labels = load_dataset_labels(dataset_name=dataset_name, data_type="VALIDATION")
    all_classes_gt = np.concatenate((training_labels, test_labels, validation_labels))
    basic_rule = []
    rule_threshold = 0.5

    for ts in all_data_combined:
        start = max(ts[9], ts[10], ts[11])
        mid = min(ts[13], ts[14], ts[15])
        end = max(ts[17], ts[18], ts[19])

        if start >= mid + rule_threshold and end >= mid + rule_threshold:
            simple_rule_guess = 0
        else:
            simple_rule_guess = 1
        basic_rule.append(simple_rule_guess)

    rule_pred_CM = metrics.confusion_matrix(basic_rule, all_classes_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=rule_pred_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.xlabel(f"Basic rule >= {rule_threshold}")
    plt.ylabel("Model Pred")
    folder_name = "BasicRule"
    make_folder(f"PyPlots/{dataset_name}/{folder_name}")

    plt.savefig(f"PyPlots/{dataset_name}/{folder_name}/rule_{rule_threshold}_pred.png")
    plt.show()

    plt.clf()
    rule_gt_CM = metrics.confusion_matrix(basic_rule, all_classes_gt)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=rule_gt_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.xlabel(f"Basic rule >= {rule_threshold}")
    plt.ylabel("Ground Truth")
    plt.savefig(f"PyPlots/{dataset_name}/{folder_name}/rule_{rule_threshold}_gt.png")

    plt.show()

    plt.clf()
    pred_gt_CM = metrics.confusion_matrix(all_classes_pred, all_classes_gt)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=pred_gt_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.xlabel("Model Pred")
    plt.ylabel("Ground Truth")
    plt.savefig(f"PyPlots/{dataset_name}/{folder_name}/pred_gt.png")
    plt.show()


def china_rule():
    dataset_name = "Chinatown"
    model_name = "Chinatown_100.keras"

    training_data = load_dataset_ts(dataset_name=dataset_name, data_type="TRAIN")
    test_data = load_dataset_ts(dataset_name=dataset_name, data_type="TEST")
    validation_data = load_dataset_ts(dataset_name=dataset_name, data_type="VALIDATION")

    all_data_combined = np.concatenate((training_data, test_data, validation_data))
    all_classes_pred = model_batch_classify(batch_of_timeseries=all_data_combined, model_name=model_name)

    training_labels = load_dataset_labels(dataset_name=dataset_name, data_type="TRAIN")
    test_labels = load_dataset_labels(dataset_name=dataset_name, data_type="TEST")
    validation_labels = load_dataset_labels(dataset_name=dataset_name, data_type="VALIDATION")
    all_classes_gt = np.concatenate((training_labels, test_labels, validation_labels))
    rule_threshold = 325
    basic_rule = []
    for ts in all_data_combined:
        y_0 = ts[0]
        y_5 = ts[5]
        diff = y_0 - y_5
        simple_rule_guess = None
        if diff >= rule_threshold:
            simple_rule_guess = 1
        else:
            simple_rule_guess = 0
        basic_rule.append(simple_rule_guess)

    rule_pred_CM = metrics.confusion_matrix(basic_rule, all_classes_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=rule_pred_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.xlabel(f"Basic rule >= {rule_threshold}")
    plt.ylabel("Model Pred")
    plt.savefig(f"PyPlots/{dataset_name}/BasicRule/rule_{rule_threshold}_pred.png")
    plt.show()

    plt.clf()
    rule_gt_CM = metrics.confusion_matrix(basic_rule, all_classes_gt)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=rule_gt_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.xlabel(f"Basic rule >= {rule_threshold}")
    plt.ylabel("Ground Truth")
    plt.savefig(f"PyPlots/{dataset_name}/BasicRule/rule_{rule_threshold}_gt.png")

    plt.show()

    plt.clf()
    pred_gt_CM = metrics.confusion_matrix(all_classes_pred, all_classes_gt)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=pred_gt_CM, display_labels=[0, 1])
    cm_display.plot()
    # Customize axis labels
    plt.xlabel("Model Pred")
    plt.ylabel("Ground Truth")
    plt.savefig(f"PyPlots/{dataset_name}/BasicRule/pred_gt.png")
    plt.show()


def calculate_simple_rule_efficenty():
    dataset_name = "Chinatown"
    model_name = "Chinatown_100.keras"

    training_data = load_dataset_ts(dataset_name=dataset_name, data_type="TRAIN")
    test_data = load_dataset_ts(dataset_name=dataset_name, data_type="TEST")
    validation_data = load_dataset_ts(dataset_name=dataset_name, data_type="VALIDATION")

    all_data_combined = np.concatenate((training_data, test_data, validation_data))
    all_classes = model_batch_classify(batch_of_timeseries=all_data_combined, model_name=model_name)

    simple_rule_TBGB = 0
    simple_rule_TBGB_diffs = []
    simple_rule_TBGR = 0
    simple_rule_TBGR_diffs = []
    simple_rule_TRGB = 0
    simple_rule_TRGB_diffs = []
    simple_rule_TRGR = 0
    simple_rule_TRGR_diffs = []
    for ts, c in zip(all_data_combined, all_classes):  # type Lost[float], int
        y_0 = ts[0]
        y_5 = ts[5]
        diff = y_0 - y_5
        simple_rule_guess = None

        if diff >= 250:
            simple_rule_guess = 0
        else:
            simple_rule_guess = 1

        if c == 0 and simple_rule_guess == 0:
            simple_rule_TBGB += 1
            simple_rule_TBGB_diffs.append(diff)
        elif c == 0 and simple_rule_guess == 1:
            simple_rule_TBGR += 1
            simple_rule_TBGR_diffs.append(diff)
        elif c == 1 and simple_rule_guess == 0:
            simple_rule_TRGB += 1
            simple_rule_TRGB_diffs.append(diff)
        elif c == 1 and simple_rule_guess == 1:
            simple_rule_TRGR += 1
            simple_rule_TRGR_diffs.append(diff)
        else:
            raise Exception("Something went wrong")

    print(
        f"simple_rule_TBGB: {simple_rule_TBGB}. Diff avg {'N/A' if len(simple_rule_TBGB_diffs) == 0 else np.average(simple_rule_TBGB_diffs)}")
    print(
        f"simple_rule_TBGR: {simple_rule_TBGR}. Diff avg {'N/A' if len(simple_rule_TBGR_diffs) == 0 else np.average(simple_rule_TBGR_diffs)}")
    print(
        f"simple_rule_TRGB: {simple_rule_TRGB}. Diff avg {'N/A' if len(simple_rule_TRGB_diffs) == 0 else np.average(simple_rule_TRGB_diffs)}")
    print(
        f"simple_rule_TRGR: {simple_rule_TRGR}. Diff avg {'N/A' if len(simple_rule_TRGR_diffs) == 0 else np.average(simple_rule_TRGR_diffs)}")


if __name__ == "__main__":
    # calculate_simple_rule_efficenty()
    # confusion_matrix()
    # IDP_rule()
    ecg_rule()
