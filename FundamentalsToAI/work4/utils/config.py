"""

@Author : Wei Mingjiang
@Time   : 2022/4/12 20:22
@File   : config.py
@Version: 0.1.0
@Content: First version.
"""
import os.path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')