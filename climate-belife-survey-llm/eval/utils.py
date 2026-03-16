import pandas as pd
import numpy as np


def discretize_score(value):
    """将0-100分数离散化为1-5级"""
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
    """四舍五入到十位"""
    if pd.isna(value):
        return np.nan
    return round(value / 10) * 10


def discretize_trees_binary(value):
    """种树二分类：没种(0) vs 种了(1-8)"""
    if pd.isna(value):
        return np.nan
    return 1 if value >= 1 else 0


def discretize_trees_ternary(value):
    """种树三分类：没种(0) vs 种了点(1-7) vs 种满(8)"""
    if pd.isna(value):
        return np.nan
    if value == 0:
        return 0  # 没种
    elif value > 4:
        return 2  # 种满
    else:
        return 1  # 种了点


def convert_share_response(response):
    """yes -> 1, no/no_social_media -> 0"""
    if response is None:
        return np.nan
    response = str(response).lower().strip()
    if response == 'yes':
        return 1
    elif response in ['no', 'no_social_media']:
        return 0
    else:
        return np.nan


def convert_share_true(value):
    """1 -> 1, 0/2 -> 0"""
    if pd.isna(value):
        return np.nan
    if value == 1:
        return 1
    elif value in [0, 2]:
        return 0
    else:
        return np.nan


def calc_true_trees(row, wept_cols):
    """计算真实种树数"""
    count = 0
    for q in ['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9']:
        col = wept_cols.get(q)
        if col and col in row.index and pd.notna(row[col]) and row[col] == 1:
            count += 1
        else:
            break
    return count
