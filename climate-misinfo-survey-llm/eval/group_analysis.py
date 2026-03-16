import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from eval.utils import round_to_decimal, discretize_score, round_ccb


def add_group_info(df_agg, df_original, group_cols):
    """Add group variable information to aggregated data"""
    cols_to_select = ['row_id'] + [col for col in group_cols.values() if col in df_original.columns]
    group_mapping = df_original[cols_to_select].copy()

    rename_dict = {v: k for k, v in group_cols.items() if v in df_original.columns}
    group_mapping = group_mapping.rename(columns=rename_dict)

    df_with_groups = df_agg.merge(group_mapping, on='row_id', how='left')

    return df_with_groups


def evaluate_continuous_by_group(df_agg, task_name, questions, group_col):
    """Evaluate continuous variables (Affect Science, Affect Action) by group"""
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


def evaluate_ccb_by_group(df_agg, group_col):
    """Evaluate CCB task by group"""
    df_task = df_agg[df_agg['task'] == 'ccb'].copy()

    if group_col not in df_task.columns:
        return None

    results = []

    for group_val in df_task[group_col].dropna().unique():
        df_group = df_task[df_task[group_col] == group_val]

        for scale_type in ['original', 'rounded']:
            all_true, all_pred = [], []

            for _, row in df_group.iterrows():
                true_val = row['true_value']
                pred_val = row['pred_value']

                if pd.isna(true_val) or pd.isna(pred_val):
                    continue

                if scale_type == 'original':
                    t, p = true_val, pred_val
                else:  # rounded
                    t, p = round_ccb(true_val), round_ccb(pred_val)

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


def evaluate_mist_by_group(df_agg, group_col):
    """Evaluate MIST binary classification + SDT metrics by group"""
    df_task = df_agg[df_agg['task'] == 'mist'].copy()

    if group_col not in df_task.columns:
        return None

    # Identify true vs false item questions
    true_qs = {f'TS{i}' for i in range(1, 6)} | {f'TD{i}' for i in range(1, 6)}
    false_qs = {f'FS{i}' for i in range(1, 6)} | {f'FD{i}' for i in range(1, 6)}

    results = []

    for group_val in df_task[group_col].dropna().unique():
        df_group = df_task[df_task[group_col] == group_val].copy()
        df_group = df_group.dropna(subset=['true_value', 'pred_value'])

        if len(df_group) == 0:
            continue

        y_true = df_group['true_value'].astype(int).values
        y_pred = df_group['pred_value'].round().astype(int).values

        row = {
            'group_var': group_col,
            'group_value': group_val,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'N': len(y_true)
        }

        # SDT metrics: signal = true statement items (TS/TD), noise = false items (FS/FD)
        df_true_items = df_group[df_group['question'].isin(true_qs)]
        df_false_items = df_group[df_group['question'].isin(false_qs)]

        if len(df_true_items) > 0 and len(df_false_items) > 0:
            hit_rate = df_true_items['pred_value'].round().mean()
            fa_rate = df_false_items['pred_value'].round().mean()

            n_true = len(df_true_items)
            n_false = len(df_false_items)
            hit_adj = np.clip(hit_rate, 0.5 / n_true, 1 - 0.5 / n_true)
            fa_adj = np.clip(fa_rate, 0.5 / n_false, 1 - 0.5 / n_false)

            row['Hit_Rate'] = hit_rate
            row['FA_Rate'] = fa_rate
            row['d_prime'] = float(norm.ppf(hit_adj) - norm.ppf(fa_adj))
            row['c_bias'] = float(-0.5 * (norm.ppf(hit_adj) + norm.ppf(fa_adj)))

        results.append(row)

    return pd.DataFrame(results) if results else None


def run_group_analysis(df_agg, df_original, group_cols):
    """Run complete group analysis"""
    print("\n" + "=" * 60)
    print("Group Analysis")
    print("=" * 60)

    df_with_groups = add_group_info(df_agg, df_original, group_cols)

    all_results = {
        'affect_science': [],
        'affect_action': [],
        'ccb': [],
        'mist': []
    }

    for group_name in group_cols.keys():
        if group_name not in df_with_groups.columns:
            continue

        print(f"\n[Analyzing by {group_name}]")

        # Affect Science
        affect_sci_by_group = evaluate_continuous_by_group(
            df_with_groups, 'affect_science', [f'S{i}' for i in range(1, 11)], group_name
        )
        if affect_sci_by_group is not None and len(affect_sci_by_group) > 0:
            all_results['affect_science'].append(affect_sci_by_group)
            print(f"    Affect Science: {len(affect_sci_by_group)} rows")

        # Affect Action
        affect_act_by_group = evaluate_continuous_by_group(
            df_with_groups, 'affect_action', [f'A{i}' for i in range(1, 11)], group_name
        )
        if affect_act_by_group is not None and len(affect_act_by_group) > 0:
            all_results['affect_action'].append(affect_act_by_group)
            print(f"    Affect Action: {len(affect_act_by_group)} rows")

        # CCB
        ccb_by_group = evaluate_ccb_by_group(df_with_groups, group_name)
        if ccb_by_group is not None and len(ccb_by_group) > 0:
            all_results['ccb'].append(ccb_by_group)
            print(f"    CCB: {len(ccb_by_group)} rows")

        # MIST
        mist_by_group = evaluate_mist_by_group(df_with_groups, group_name)
        if mist_by_group is not None and len(mist_by_group) > 0:
            all_results['mist'].append(mist_by_group)
            print(f"    MIST: {len(mist_by_group)} rows")

    return all_results
