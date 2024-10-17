from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def parse_tsv(dataset_name: str) -> np.ndarray:
    """
    Parse the data from TSV file into a Dataframe, and transform it into a numpy array.
    :param dataset_name:
    :return:
    """
    folder = "dataSet/UCR_DATA/"
    file_location = folder + dataset_name
    array_2d = pd.read_csv(file_location, delimiter=r"\s+", header=None, dtype=float)
    array_2d = np.array(array_2d, dtype=float)

    return array_2d


import random


def convert_data(dataset, shuffle=False, random_state=42):
    """
    Split the UCR format into train, valid and test data.
    :param shuffle:
    :param random_state:
    :param dataset:
    :return:
    """

    # There is a choice to be made here. Should I use train from UCR and as train, and split test
    # into test and validate.
    # I will for now do train -> train, and test -> validation, test.
    # train_perc = 0.4
    # val_perc = 0.3
    test_perc = 0.7
    train = parse_tsv(dataset + "_TRAIN.tsv")
    random.seed(42)
    if shuffle:
        np.random.shuffle(train)

    np.save(f"dataSet/{dataset}_TRAIN", train)
    test_org = parse_tsv(dataset + "_TEST.tsv")
    val, test = train_test_split(test_org,
                                 test_size=test_perc, random_state=random_state)
    np.save(f"dataSet/{dataset}_VALIDATION", val)
    np.save(f"dataSet/{dataset}_TEST", test)

    print(val, test)
    print(len(val), len(test))


if __name__ == '__main__':
    convert_data("ItalyPowerDemand")
    convert_data("Chinatown")
    convert_data("ECG200")
    convert_data("Charging")
