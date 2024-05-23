import tensorflow as tf
import numpy as np

MODELS = {}


def load_keras_model(model_name):
    global MODELS
    if model_name not in MODELS:
        print(model_name)
        keras_model = tf.keras.models.load_model(f"models/kerasModels/{model_name}")
        keras_model.trainable = False
        MODELS[model_name] = keras_model
    return MODELS[model_name]


def model_classify(model_name, time_series):
    model = load_keras_model(model_name)
    input_shape = model.input_shape[1:]  # Get Shape
    reshaped_input = np.array(time_series).reshape(input_shape)
    time_series = np.array([reshaped_input])  # (batchsize=1,(inputshape))
    predictions = model.predict(time_series)
    class_pred = np.argmax(predictions, axis=1)[0]  # Extract batch index 0
    return class_pred


def model_batch_classify(model_name, batch_of_timeseries):
    model = load_keras_model(model_name)

    # The Batch should already be correct format
    input_shape = model.input_shape[1:]  # Get Shape
    batch_of_timeseries = [np.array(timeseries).reshape(input_shape) for timeseries in batch_of_timeseries]
    batch_of_timeseries = np.array(batch_of_timeseries)
    predictions = model.predict(batch_of_timeseries, verbose=False)
    class_pred = [np.argmax(prediction) for prediction in predictions]
    return class_pred


def model_confidence(dataset, timeseries):
    model = load_keras_model(dataset)
    input_shape = model.input_shape[1:]  # Get Shape
    reshaped_input = timeseries.reshape(input_shape)
    timeseries = np.array([reshaped_input])  # (batchsize=1,(inputshape))
    predictions = model.predict(timeseries)
    confidence = np.max(predictions)
    return confidence


def batch_confidence(model_name, batch_of_timeseries):
    model = load_keras_model(model_name)
    # The Batch should already be correct format
    input_shape = model.input_shape[1:]  # Get Shape
    batch_of_timeseries = [np.array(timeseries).reshape(input_shape) for timeseries in batch_of_timeseries]
    batch_of_timeseries = np.array(batch_of_timeseries)
    predictions = model.predict(batch_of_timeseries)
    confidence_pred = np.array([np.max(prediction) for prediction in predictions])
    return confidence_pred
