# -*- coding: utf-8 -*-

from utils import *

class DNN:
    def __init__(self, input_shape, learning_rate=0.005):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        # Input layer
        inputs = keras.Input(shape=(self.input_shape,), name="input")

        # Hidden layers
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)

        # Output layers
        x = layers.Dense(4, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid', name='stress')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])

        return model

    def fit(self, xtra_ac, ytra_ac, xval_ac, yval_ac, checkpoint_path, epochs=30, batch_size=32):
        """Fits the model to the training data."""
        mcp_save = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
        callbacks = [mcp_save]
        self.model.fit(xtra_ac, ytra_ac,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(xval_ac, yval_ac),
                       callbacks=callbacks)

    def evaluate(self, xtest_ac, ytest_ac):
        """Evaluates the model on test data."""
        accuracy = self.model.evaluate(xtest_ac, ytest_ac)[1]
        return accuracy

    def predict(self, xtest_ac):
        """Makes predictions on test data."""
        pred_output = self.model.predict(xtest_ac)
        #y_pred_a = np.where(pred_output > 0.5, 1, 0)
        return pred_output

    def get_model(self):
        """Returns the compiled model."""
        return self.model
