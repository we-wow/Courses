"""

@Author : Wei Mingjiang
@Time   : 2022/3/28 15:25
@File   : load_forecasting.py
@Version: 0.1.0
@Content: First version.
"""
import datetime
import os.path

import keras.layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.model_selection import train_test_split
from utils.config import DATA_DIR, BASE_DIR


class LoadForecasting:
    def __init__(self, data_file="2016年电工数学建模竞赛负荷预测数据集.txt", x_dim=5, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = [64, 128, 64]
        self.data_path = os.path.join(DATA_DIR, data_file)
        self.data = self._read_data()
        self.x_dim = x_dim
        self.model = self.build_model(hidden_layers)

    def _read_data(self):
        _data = []
        with open(self.data_path, 'r') as f:
            f.readline()
            for line in f.readlines():
                if not line or "NaN" in line:
                    continue
                simple = line.strip().split()
                if len(simple) != 7:
                    continue
                time = datetime.datetime.strptime(simple[0], "%Y%m%d")
                days_in_year = (time - datetime.datetime(time.year - 1, 12, 31)).days
                _data.append([days_in_year / 365, ] + [float(d) for d in simple[1: ]])
        _data = np.array(_data)
        _data[:, -1] *= 1e-5
        return np.array(_data)

    def preprocess_kernel_pca(self):
        X = self.data[:, :-1]
        k_pca = KernelPCA(kernel='linear', n_components=self.x_dim)
        _X = k_pca.fit_transform(X)
        # pca = PCA(n_components=self.x_dim)
        # pca.fit_transform(X)
        # print(sum(pca.explained_variance_ratio_))
        return _X

    def build_model(self, hidden_layers):
        inputs = keras.layers.Input(shape=(self.x_dim, ))
        x = inputs
        for lay in hidden_layers:
            x = keras.layers.Dense(lay, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='relu')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='MSE',
            # metrics='mse',
        )
        return model

    def train(self, epochs=200):
        if self.x_dim == self.data.shape[1] - 1:
            new_x = self.data[:, :-1]
        else:
            new_x = self.preprocess_kernel_pca()

        y = self.data[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2, random_state=0)

        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 validation_split=0.1,
                                 batch_size=32,
                                 verbose=0
                                 )
        self.model.save_weights(os.path.join(BASE_DIR, "model/model.h5"))
        plt.plot(history.epoch, history.history['loss'])
        plt.plot(history.epoch, history.history['val_loss'])
        plt.legend(['train_loss', 'test_loss'])
        plt.show()
        return X_test, y_test
        
    def predict(self, x_test, y_test):
        y_pre = self.model.predict(x_test)
        y_pre = y_pre.reshape(y_pre.shape[0])
        acc = 1 - np.mean(np.abs(y_pre - y_test) / y_test)
        print(f"Predict Accuracy: {acc * 100:.2f}%")
        return y_pre

        
if __name__ == '__main__':
    lf = LoadForecasting(hidden_layers=[128, 128], x_dim=3)
    lf.preprocess_kernel_pca()
    v_x, v_y = lf.train()
    y_p = lf.predict(v_x, v_y)
    