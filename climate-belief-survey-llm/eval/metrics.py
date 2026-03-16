import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from eval.utils import round_to_decimal, discretize_score, discretize_trees_binary, discretize_trees_ternary


def evaluate_continuous(df_raw, df_agg, task_name, questions):
    """
    评估连续变量（Belief, Policy）
    返回：逐题详情（聚合后）、按scale按题目的指标
    """
    print(f"\n[Evaluating {task_name}]")

    df_task = df_agg[df_agg['task'] == task_name].copy()

    # ========== 逐题详情 ==========
    detail_rows = []
    for _, row in df_task.iterrows():
        true_val = row['true_value']
        pred_val = row['pred_value']

        if pd.isna(true_val):
            continue

        detail_rows.append({
            'row_id': row['row_id'],
            'question': row['question'],
            'true_original': true_val,
            'pred_original': pred_val,
            'true_decimal': round_to_decimal(true_val),
            'pred_decimal': round_to_decimal(pred_val) if pd.notna(pred_val) else np.nan,
            'true_discrete': discretize_score(true_val),
            'pred_discrete': discretize_score(pred_val) if pd.notna(pred_val) else np.nan,
            'error': pred_val - true_val if pd.notna(pred_val) else np.nan,
            'abs_error': abs(pred_val - true_val) if pd.notna(pred_val) else np.nan
        })

    df_detail = pd.DataFrame(detail_rows)

    # ========== 按题目计算指标 ==========
    metrics_rows = []

    for scale_type in ['original', 'decimal', 'discrete']:
        all_true, all_pred = [], []

        for q in questions:
            q_data = df_detail[df_detail['question'] == q].copy()
            q_data = q_data.dropna(subset=[f'true_{scale_type}', f'pred_{scale_type}'])

            if len(q_data) == 0:
                continue

            y_true = q_data[f'true_{scale_type}'].values
            y_pred = q_data[f'pred_{scale_type}'].values

            mean_error = np.mean(y_pred - y_true)
            mae = np.mean(np.abs(y_pred - y_true))

            metrics_rows.append({
                'scale': scale_type,
                'question': q,
                'Mean_Error': mean_error,
                'MAE': mae,
                'N': len(y_true)
            })

            all_true.extend(y_true)
            all_pred.extend(y_pred)

        if len(all_true) > 0:
            mean_error = np.mean(np.array(all_pred) - np.array(all_true))
            mae = np.mean(np.abs(np.array(all_pred) - np.array(all_true)))

            metrics_rows.append({
                'scale': scale_type,
                'question': 'OVERALL',
                'Mean_Error': mean_error,
                'MAE': mae,
                'N': len(all_true)
            })

    df_metrics = pd.DataFrame(metrics_rows)

    overall = df_metrics[(df_metrics['scale'] == 'original') & (df_metrics['question'] == 'OVERALL')]
    if len(overall) > 0:
        m = overall.iloc[0]
        print(f"    Original: Mean_Error={m['Mean_Error']:.2f}, MAE={m['MAE']:.2f}, N={m['N']}")

    return df_detail, df_metrics


def evaluate_share(df_raw, df_agg):
    """评估Share分类任务"""
    print(f"\n[Evaluating Share]")

    df_task = df_agg[df_agg['task'] == 'share'].copy()

    detail_rows = []
    for _, row in df_task.iterrows():
        true_val = row['true_value']
        pred_val = row['pred_value']

        if pd.isna(true_val):
            continue

        pred_round = round(pred_val) if pd.notna(pred_val) else np.nan
        correct = (true_val == pred_round) if pd.notna(pred_round) else np.nan

        detail_rows.append({
            'row_id': row['row_id'],
            'true_share': int(true_val),
            'pred_share_mean': pred_val,
            'pred_share_round': pred_round,
            'correct': 1 if correct else 0
        })

    df_detail = pd.DataFrame(detail_rows)

    if len(df_detail) == 0:
        return df_detail, None, None

    df_clean = df_detail.dropna(subset=['true_share', 'pred_share_round'])

    if len(df_clean) == 0:
        return df_detail, None, None

    y_true = df_clean['true_share'].astype(int).values
    y_pred = df_clean['pred_share_round'].astype(int).values

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'N': len(y_true)
    }

    cm = confusion_matrix(y_true, y_pred)

    print(f"    Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}, N: {metrics['N']}")

    return df_detail, metrics, cm


def evaluate_trees(df_raw, df_agg):
    """
    评估种树任务
    包括：连续值、二分类（种了/没种）、三分类（没种/种了点/种满）
    """
    print(f"\n[Evaluating Trees]")

    df_task = df_agg[df_agg['task'] == 'trees'].copy()

    detail_rows = []
    for _, row in df_task.iterrows():
        true_val = row['true_value']
        pred_val = row['pred_value']

        pred_round = round(pred_val) if pd.notna(pred_val) else np.nan

        detail_rows.append({
            'row_id': row['row_id'],
            'true_trees': true_val,
            'pred_trees_mean': pred_val,
            'pred_trees_round': pred_round,
            'error': pred_val - true_val if pd.notna(pred_val) and pd.notna(true_val) else np.nan,
            'abs_error': abs(pred_val - true_val) if pd.notna(pred_val) and pd.notna(true_val) else np.nan,
            'true_binary': discretize_trees_binary(true_val),
            'pred_binary': discretize_trees_binary(pred_round),
            'true_ternary': discretize_trees_ternary(true_val),
            'pred_ternary': discretize_trees_ternary(pred_round)
        })

    df_detail = pd.DataFrame(detail_rows)

    if len(df_detail) == 0:
        return df_detail, None

    results = {}

    # 1. 连续值指标
    df_valid = df_detail.dropna(subset=['true_trees', 'pred_trees_mean'])
    if len(df_valid) > 0:
        y_true = df_valid['true_trees'].values
        y_pred = df_valid['pred_trees_mean'].values

        results['continuous'] = {
            'Mean_Error': np.mean(y_pred - y_true),
            'MAE': np.mean(np.abs(y_pred - y_true)),
            'N': len(y_true)
        }
        print(
            f"    Continuous: Mean_Error={results['continuous']['Mean_Error']:.2f}, MAE={results['continuous']['MAE']:.2f}")

    # 2. 二分类指标（种了 vs 没种）
    df_binary = df_detail.dropna(subset=['true_binary', 'pred_binary'])
    if len(df_binary) > 0:
        y_true = df_binary['true_binary'].astype(int).values
        y_pred = df_binary['pred_binary'].astype(int).values

        results['binary'] = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'N': len(y_true)
        }
        results['binary_cm'] = confusion_matrix(y_true, y_pred)
        print(
            f"    Binary (planted vs not): Accuracy={results['binary']['Accuracy']:.4f}, F1={results['binary']['F1']:.4f}")

    # 3. 三分类指标（没种/种了点/种满）
    df_ternary = df_detail.dropna(subset=['true_ternary', 'pred_ternary'])
    if len(df_ternary) > 0:
        y_true = df_ternary['true_ternary'].astype(int).values
        y_pred = df_ternary['pred_ternary'].astype(int).values

        results['ternary'] = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'N': len(y_true)
        }
        results['ternary_cm'] = confusion_matrix(y_true, y_pred)
        print(
            f"    Ternary (none/some/full): Accuracy={results['ternary']['Accuracy']:.4f}, F1={results['ternary']['F1']:.4f}")

    # 4. 分布统计
    df_dist = df_detail.dropna(subset=['true_trees', 'pred_trees_round'])
    if len(df_dist) > 0:
        distribution = {'true': {}, 'pred': {}}
        for i in range(9):
            distribution['true'][i] = int((df_dist['true_trees'] == i).sum())
            distribution['pred'][i] = int((df_dist['pred_trees_round'] == i).sum())
        results['distribution'] = distribution

    return df_detail, results


def evaluate_correlation(df_agg, task_name, questions):
    """
    Compute Pearson and Spearman correlations between LLM predictions and true values.
    Returns: DataFrame with columns [question, pearson_r, pearson_p, spearman_r, spearman_p, N]
    """
    df_task = df_agg[df_agg['task'] == task_name].copy()

    corr_rows = []
    all_true, all_pred = [], []

    for q in questions:
        q_data = df_task[df_task['question'] == q].dropna(subset=['true_value', 'pred_value'])
        if len(q_data) < 3:
            continue

        y_true = q_data['true_value'].values
        y_pred = q_data['pred_value'].values

        try:
            pr, pp = pearsonr(y_true, y_pred)
        except Exception:
            pr, pp = np.nan, np.nan
        try:
            sr, sp = spearmanr(y_true, y_pred)
        except Exception:
            sr, sp = np.nan, np.nan

        corr_rows.append({
            'question': q,
            'pearson_r': round(pr, 4) if not np.isnan(pr) else np.nan,
            'pearson_p': round(pp, 4) if not np.isnan(pp) else np.nan,
            'spearman_r': round(sr, 4) if not np.isnan(sr) else np.nan,
            'spearman_p': round(sp, 4) if not np.isnan(sp) else np.nan,
            'N': len(y_true)
        })

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    # OVERALL
    if len(all_true) >= 3:
        try:
            pr, pp = pearsonr(all_true, all_pred)
        except Exception:
            pr, pp = np.nan, np.nan
        try:
            sr, sp = spearmanr(all_true, all_pred)
        except Exception:
            sr, sp = np.nan, np.nan

        corr_rows.append({
            'question': 'OVERALL',
            'pearson_r': round(pr, 4) if not np.isnan(pr) else np.nan,
            'pearson_p': round(pp, 4) if not np.isnan(pp) else np.nan,
            'spearman_r': round(sr, 4) if not np.isnan(sr) else np.nan,
            'spearman_p': round(sp, 4) if not np.isnan(sp) else np.nan,
            'N': len(all_true)
        })

    df_corr = pd.DataFrame(corr_rows)
    if len(df_corr) > 0:
        overall = df_corr[df_corr['question'] == 'OVERALL']
        if len(overall) > 0:
            m = overall.iloc[0]
            print(f"    Correlation OVERALL: pearson_r={m['pearson_r']}, spearman_r={m['spearman_r']}, N={m['N']}")
    return df_corr
