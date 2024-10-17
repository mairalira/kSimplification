import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical


class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, dataset_name, epochs=500, verbose=False, build=True):
        self.output_directory = output_directory
        self.dataset_name = dataset_name
        self.epochs = epochs
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
        return

    def build_model(self, input_shape, nb_classes):
        input_shape = (input_shape[0], 1)
        print(input_shape)
        input_layer = keras.layers.Input(input_shape)
        print(input_shape, nb_classes)
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
        #              metrics=['accuracy'])

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = "models/CheckPoint/" + str(self.dataset_name) + '.keras'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_test, y_test, verbose=True):
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        batch_size = 16
        nb_epochs = self.epochs  # 2000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        hist = self.model.fit(x_train, to_categorical(y_train), batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, callbacks=self.callbacks,
                              validation_data=(x_test, to_categorical(y_test)),
                              class_weight=class_weight_dict
                              )

        self.keras_save()

        # Plotting the training and validation loss
        if verbose:
            plt.figure()
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title(f'{self.dataset_name} model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper right')
            plt.show()

        # model = keras.models.load_model("Blackbox_classifier_FCN/" + str(self.dataset_name) + '_best_model.hdf5')

    def predict(self, x_test):
        model_path = "Blackbox_classifier_FCN/model" + str(self.dataset_name) + '.keras'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def keras_save(self):
        self.model.save('models/kerasModels/' + str(self.dataset_name) + "_" + str(self.epochs) + '.keras')
