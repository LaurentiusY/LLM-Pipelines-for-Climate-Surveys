import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from eval.utils import round_to_decimal, discretize_score, discretize_trees_binary, discretize_trees_ternary


def add_group_info(df_agg, df_original, group_cols):
    """将分组变量信息添加到聚合数据中"""
    cols_to_select = ['row_id'] + [col for col in group_cols.values() if col in df_original.columns]
    group_mapping = df_original[cols_to_select].copy()

    rename_dict = {v: k for k, v in group_cols.items() if v in df_original.columns}
    group_mapping = group_mapping.rename(columns=rename_dict)

    df_with_groups = df_agg.merge(group_mapping, on='row_id', how='left')

    return df_with_groups


def evaluate_continuous_by_group(df_agg, task_name, questions, group_col):
    """按分组评估连续变量（Belief, Policy）"""
    df_task = df_agg[df_agg['task'] == task_name].copy()

    if group_col not in df_task.columns:
        return None

    results = []

    for group_val in df_task[group_col].dropna().unique():
        df_group = df_task[df_task[group_col] == group_val]

        for scale_type in ['original', 'decimal', 'discrete']:
            all_true, all_pred = [], []

            for _, row in df_group.iterrows():
                true_val = row['true_value']
                pred_val = row['pred_value']

                if pd.isna(true_val) or pd.isna(pred_val):
                    continue

                if scale_type == 'original':
                    t, p = true_val, pred_val
                elif scale_type == 'decimal':
                    t, p = round_to_decimal(true_val), round_to_decimal(pred_val)
                else:  # discrete
                    t, p = discretize_score(true_val), discretize_score(pred_val)

                all_true.append(t)
                all_pred.append(p)

            if len(all_true) > 0:
                mean_error = np.mean(np.array(all_pred) - np.array(all_true))
                mae = np.mean(np.abs(np.array(all_pred) - np.array(all_true)))

                results.append({
                    'group_var': group_col,
                    'group_value': group_val,
                    'scale': scale_type,
                    'Mean_Error': mean_error,
                    'MAE': mae,
                    'N': len(all_true)
                })

    return pd.DataFrame(results) if results else None


def evaluate_share_by_group(df_agg, group_col):
    """按分组评估Share分类任务"""
    df_task = df_agg[df_agg['task'] == 'share'].copy()

    if group_col not in df_task.columns:
        return None

    results = []

    for group_val in df_task[group_col].dropna().unique():
        df_group = df_task[df_task[group_col] == group_val].copy()
        df_group = df_group.dropna(subset=['true_value', 'pred_value'])

        if len(df_group) == 0:
            continue

        y_true = df_group['true_value'].astype(int).values
        y_pred = df_group['pred_value'].round().astype(int).values

        results.append({
            'group_var': group_col,
            'group_value': group_val,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'N': len(y_true)
        })

    return pd.DataFrame(results) if results else None


def evaluate_trees_by_group(df_agg, group_col):
    """按分组评估种树任务"""
    df_task = df_agg[df_agg['task'] == 'trees'].copy()

    if group_col not in df_task.columns:
        return None

    results = []

    for group_val in df_task[group_col].dropna().unique():
        df_group = df_task[df_task[group_col] == group_val].copy()
        df_valid = df_group.dropna(subset=['true_value', 'pred_value'])

        if len(df_valid) == 0:
            continue

        y_true = df_valid['true_value'].values
        y_pred = df_valid['pred_value'].values
        y_pred_round = np.round(y_pred)

        result = {
            'group_var': group_col,
            'group_value': group_val,
            'Mean_Error': np.mean(y_pred - y_true),
            'MAE': np.mean(np.abs(y_pred - y_true)),
            'N': len(y_true)
        }

        true_binary = np.array([discretize_trees_binary(v) for v in y_true])
        pred_binary = np.array([discretize_trees_binary(v) for v in y_pred_round])
        valid_mask = ~(np.isnan(true_binary) | np.isnan(pred_binary))

        if valid_mask.sum() > 0:
            tb, pb = true_binary[valid_mask].astype(int), pred_binary[valid_mask].astype(int)
            result['Binary_Accuracy'] = accuracy_score(tb, pb)
            result['Binary_F1'] = f1_score(tb, pb, zero_division=0)

        true_ternary = np.array([discretize_trees_ternary(v) for v in y_true])
        pred_ternary = np.array([discretize_trees_ternary(v) for v in y_pred_round])
        valid_mask = ~(np.isnan(true_ternary) | np.isnan(pred_ternary))

        if valid_mask.sum() > 0:
            tt, pt = true_ternary[valid_mask].astype(int), pred_ternary[valid_mask].astype(int)
            result['Ternary_Accuracy'] = accuracy_score(tt, pt)
            result['Ternary_F1'] = f1_score(tt, pt, average='macro', zero_division=0)

        results.append(result)

    return pd.DataFrame(results) if results else None


def run_group_analysis(df_agg, df_original, group_cols):
    """运行完整的分组分析"""
    print("\n" + "=" * 60)
    print("Group Analysis")
    print("=" * 60)

    df_with_groups = add_group_info(df_agg, df_original, group_cols)

    all_results = {
        'belief': [],
        'policy': [],
        'share': [],
        'trees': []
    }

    for group_name in group_cols.keys():
        if group_name not in df_with_groups.columns:
            continue

        print(f"\n[Analyzing by {group_name}]")

        belief_by_group = evaluate_continuous_by_group(
            df_with_groups, 'belief', [f'Q{i}' for i in range(1, 5)], group_name
        )
        if belief_by_group is not None and len(belief_by_group) > 0:
            all_results['belief'].append(belief_by_group)
            print(f"    Belief: {len(belief_by_group)} rows")

        policy_by_group = evaluate_continuous_by_group(
            df_with_groups, 'policy', [f'Q{i}' for i in range(1, 10)], group_name
        )
        if policy_by_group is not None and len(policy_by_group) > 0:
            all_results['policy'].append(policy_by_group)
            print(f"    Policy: {len(policy_by_group)} rows")

        share_by_group = evaluate_share_by_group(df_with_groups, group_name)
        if share_by_group is not None and len(share_by_group) > 0:
            all_results['share'].append(share_by_group)
            print(f"    Share: {len(share_by_group)} rows")

        trees_by_group = evaluate_trees_by_group(df_with_groups, group_name)
        if trees_by_group is not None and len(trees_by_group) > 0:
            all_results['trees'].append(trees_by_group)
            print(f"    Trees: {len(trees_by_group)} rows")

    return all_results
