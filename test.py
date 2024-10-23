from models.loadModel import model_batch_classify
import numpy as np

#model_name = "Charging.keras" #Original code
model_name = 'Chinatown.keras'

ts = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
ts_batch = np.array([ts.copy() for _ in range(1000000)])
print(len(ts_batch))
preds = model_batch_classify(model_name=model_name, batch_of_timeseries=ts_batch)
print(len(preds))
print(sum(preds))
