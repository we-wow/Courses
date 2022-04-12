"""
Feature selection based on random forest regression.
@Author : Wei Mingjiang
@Time   : 2022/4/12 14:02
@File   : problem1.py
@Version: 0.1.0
@Content: First version.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

from utils.read_pickle_file import read_pickle


training_data = read_pickle('Molecular_Descriptor_training_header')
training_label = read_pickle('ERα_activity_training_header')

header = training_data[0, 1:]
X = training_data[1:, 1:]
y = training_label[1:, 2]

max_depth = 10
max_features = 20
rf = RandomForestRegressor(max_depth=max_depth, max_features=max_features)
rf.fit(X, y)

sorted_importance = sorted(zip(map(lambda x: round(x, 6), rf.feature_importances_),
                               [_ + 1 for _ in range(X.shape[1])]), reverse=True)
res = [feature[1] for feature in sorted_importance[:20]]
print(sorted_importance)
print(sorted(res))
