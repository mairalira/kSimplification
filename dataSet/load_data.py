import numpy as np
import pandas as pd
def load_dataset(dataset_name):
    folder = "dataSet/"
    file_location = folder + dataset_name
    array_2d = pd.read_csv(file_location, header=None)
    array_2d = np.array(array_2d, dtype=float)

    return array_2d


def test():
    load_dataset("Chinatown.csv")


if __name__ == "__main__":
    test()



