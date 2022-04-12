"""
Class model.
@Author : Wei Mingjiang
@Time   : 2022/4/12 14:02
@File   : problem1.py
@Version: 0.1.0
@Content: First version.
"""
import os.path
import tensorflow.keras


class Model:
    def __init__(self,
                 input_length,
                 name="model_problem2",
                 loss_func='mae',
                 optimizer='adam',
                 weights_path=None
                 ):
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.name = name
        self._input_length = input_length
        self.model = self._build_model()
        self._weights_path = weights_path
        if self._weights_path is not None:
            if os.path.exists(self._weights_path):
                self.model.load_weights(self._weights_path)
            else:
                raise FileExistsError(f"Path {self._weights_path} does not exist.")

    def _build_model(self):
        input_layer = tensorflow.keras.Input(shape=(self._input_length,))
        x = tensorflow.keras.layers.Dense(64, activation='relu')(input_layer)
        x = tensorflow.keras.layers.Dense(64, activation='relu')(x)
        x = tensorflow.keras.layers.Dense(32, activation='relu')(x)
        output_layer = tensorflow.keras.layers.Dense(1)(x)

        model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer, name=self.name)
        model.compile(optimizer=self.optimizer, loss=self.loss_func)
        return model

    def predict(self, _x):
        return self.model.predict(_x)


