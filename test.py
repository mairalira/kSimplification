from models.loadModel import model_batch_classify
import numpy as np

from dataSet.load_data import load_dataset_labels

dataset_name = "ItalyPowerDemand"

all_labels = load_dataset_labels(dataset_name, data_type="TEST")

zero = 0
one = 0
for l in all_labels:
    if l == 0:
        zero += 1
    elif l == 1:
        one += 1
    else:
        print("WHAT???", l)

print(zero, one)
