"""
Problem. 2
@Author : Wei Mingjiang
@Time   : 2022/4/12 14:02
@File   : problem1.py
@Version: 0.1.0
@Content: First version.
"""

import tensorflow.keras as tk
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.model_selection import train_test_split

import base_model
from utils.read_pickle_file import read_pickle
from utils.config import BASE_DIR

training_data = read_pickle('Molecular_Descriptor_training')
training_label = read_pickle('ERα_activity_training')


X = training_data[:, 1:].astype(np.float32)
y = training_label[:, 2].astype(np.float32)

# index start from  1
chosen_features = [5, 12, 36, 40, 96, 120, 239, 288, 358, 407, 413, 477, 526, 530, 532, 588, 647, 653, 660, 674]
X = X.take(chosen_features, 1)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.01, random_state=0)


loss_func = 'mae'
optimizer = 'adam'
model = base_model.Model(
    input_length=len(chosen_features),
    loss_func=loss_func,
    optimizer=optimizer,
)

# Train
batch_size = 32
epochs = 200
validation_split = 0.2
validation_freq = 1
history = model.model.fit(
    train_X, train_y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split,
    validation_freq=validation_freq,
    verbose=2)

plt.plot(history.epoch, history.history['loss'])
plt.plot(history.epoch, history.history['val_loss'])
plt.legend(['train', 'test'])
plt.show()
if input('Save?[y/n]:') not in ['n', 'N']:
    # Set path
    result_path = os.path.join(BASE_DIR, "result", "problem2")
    # Save Weights
    max_id = 0
    for each in os.listdir(result_path):
        folder_id = int(each.split('_')[0])
        if max_id < folder_id:
            max_id = folder_id
    t = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    present_model_files_path = os.path.join(BASE_DIR, "result", "problem2", f"{max_id + 1}_{t}")
    history_log_path = os.path.join(present_model_files_path, "train_history.log")
    figure_path = os.path.join(present_model_files_path, "history.png")
    log_path = os.path.join(present_model_files_path, "problem2_analysis.log")

    os.mkdir(present_model_files_path)
    weights_path = os.path.join(present_model_files_path, "model.h5")
    model.model.save(weights_path)

    # Save history
    with open(history_log_path, 'w', encoding='utf-8') as f:
        f.write("epoch      loss  val_loss\n")
        for i in history.epoch:
            f.write(f"{i:5d}  {history.history['loss'][i]:8.6f}  {history.history['val_loss'][i]:8.6f}\n")

    # Save figure
    plt.close()
    plt.plot(history.epoch, history.history['loss'])
    plt.plot(history.epoch, history.history['val_loss'])
    plt.legend(['train', 'test'])
    plt.savefig(figure_path)

    # Logs
    pre_y = model.predict(test_X)
    test_mse = model.model.evaluate(test_X, test_y, batch_size=32)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=============Model Parameters================\n")
        f.write(f"Optimizer：{optimizer}\nLoss function：{loss_func}\n")


    def log_model_summary(text):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')


    model.model.summary(print_fn=log_model_summary)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\n=============Training Parameters================\n")
        f.write(f"Weights File Path: {weights_path}\n")
        f.write(f"History Figure File Path: {figure_path}\n")
        f.write(f" Features: {chosen_features}\n")
        f.write(
            f" batch_size: {batch_size}\n epochs:{epochs}\n validation_split:{validation_split}\n validation_freq:{validation_freq}\n")
        f.write("\n=============Evaluate Results================\n")
        f.write(f"MSE: {test_mse:7.5f}\n")
        f.write("predict  truth value     err(%)\n")
        for i in range(len(pre_y)):
            f.write(f"{pre_y[i][0]:7.5f}  {test_y[i]:11.5f}  {abs(pre_y[i][0] - test_y[i]) / test_y[i] * 100:9.4f}\n")