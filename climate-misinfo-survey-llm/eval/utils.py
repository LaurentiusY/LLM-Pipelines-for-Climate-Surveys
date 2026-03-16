import pandas as pd
import numpy as np


def discretize_score(value):
    """Discretize 0-100 score to 1-5 levels"""
    if pd.isna(value):
        return np.nan
    if value <= 20:
        return 1
    elif value <= 40:
        return 2
    elif value <= 60:
        return 3
    elif value <= 80:
        return 4
    else:
        return 5


def round_to_decimal(value):
    """Round to nearest 10"""
    if pd.isna(value):
        return np.nan
    return round(value / 10) * 10


def round_ccb(value):
    """Round CCB 1-5 to nearest integer"""
    if pd.isna(value):
        return np.nan
    return max(1, min(5, round(value)))


def discretize_affect_ternary(value):
    """Collapse 0-100 affect score into 3 classes: negative(0-33)=0, neutral(34-66)=1, positive(67-100)=2"""
    if pd.isna(value):
        return np.nan
    if value <= 33:
        return 0
    elif value <= 66:
        return 1
    else:
        return 2


def convert_mist_response(response):
    """'true' -> 1, 'false' -> 0"""
    if response is None:
        return np.nan
    response = str(response).lower().strip()
    if response == 'true':
        return 1
    elif response == 'false':
        return 0
    else:
        return np.nan
