import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dataSet.load_data import load_dataset_ts, load_dataset_labels
from train_models.FCN_model import Classifier_FCN


def load_model(dataset):
    model = keras.models.load_model('Blackbox_classifier_FCN/' + str(dataset) + '_best_model.hdf5')
    return model


def train_model(dataset: str, epochs=500, verbose=True):
    X_train = load_dataset_ts(dataset_name=dataset, data_type="TRAIN")
    y_train = load_dataset_labels(dataset_name=dataset, data_type="TRAIN")
    input_shape = X_train.shape[1:]
    nb_classes = len(np.unique(y_train))
    dataset_name = str(dataset)

    fcn = Classifier_FCN(output_directory=os.getcwd(), input_shape=input_shape, nb_classes=nb_classes,
                         dataset_name=dataset_name, verbose=verbose, epochs=epochs)

    X_test = load_dataset_ts(dataset, data_type="TEST")
    y_test = load_dataset_labels(dataset, data_type="TEST")
    fcn.fit(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)

    fcn.keras_save()


def test():
    datasets = ["Chinatown", "ItalyPowerDemand"]
    for dataset in datasets:
        train_model(dataset, epochs=100)


if __name__ == '__main__':
    test()
