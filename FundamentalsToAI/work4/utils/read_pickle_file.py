"""
Read pickle file.
@Author : Wei Mingjiang
@Time   : 2022/4/12 14:02
@File   : problem1.py
@Version: 0.1.0
@Content: First version.
"""
import pickle
from .config import DATA_DIR


def read_pickle(file_name: str):
    with open(f"{DATA_DIR}/pickle/{file_name}", 'rb') as f:
        variable = pickle.load(f)
    return variable
