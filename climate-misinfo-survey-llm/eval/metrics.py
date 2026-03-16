import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
from eval.utils import round_to_decimal, discretize_score, round_ccb, discretize_affect_ternary


def evaluate_continuous(df_raw, df_agg, task_name, questions):
    """
    Evaluate continuous variables (Affect Science, Affect Action) on 0-100 scale.
    Returns: per-question detail (aggregated), metrics by scale and question.
    """
    print(f"\n[Evaluating {task_name}]")

    df_task = df_agg[df_agg['task'] == task_name].copy()

    # ========== Per-question detail ==========
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

    # ========== Metrics by question ==========
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


def evaluate_ccb(df_raw, df_agg):
    """
    Evaluate CCB task (1-5 continuous scale).
    Returns: detail, metrics at original and rounded (integer) scales.
    """
    print(f"\n[Evaluating CCB]")

    df_task = df_agg[df_agg['task'] == 'ccb'].copy()

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
            'true_rounded': round_ccb(true_val),
            'pred_rounded': round_ccb(pred_val) if pd.notna(pred_val) else np.nan,
            'error': pred_val - true_val if pd.notna(pred_val) else np.nan,
            'abs_error': abs(pred_val - true_val) if pd.notna(pred_val) else np.nan
        })

    df_detail = pd.DataFrame(detail_rows)

    metrics_rows = []
    questions = ['Q1', 'Q2', 'Q3']

    for scale_type in ['original', 'rounded']:
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


def evaluate_mist(df_raw, df_agg):
    """
    Evaluate MIST binary classification task.
    Returns: detail, metrics (accuracy/precision/recall/F1 per category and overall),
             confusion matrix, SDT metrics (d-prime, c-bias).

    MIST items:
      - TS1-5, TD1-5: True items (human responded 1=said true, 0=said false)
      - FS1-5, FD1-5: False items (human responded 1=said true, 0=said false)
    We compare LLM response against human response.
    """
    print(f"\n[Evaluating MIST]")

    df_task = df_agg[df_agg['task'] == 'mist'].copy()

    detail_rows = []
    for _, row in df_task.iterrows():
        true_val = row['true_value']
        pred_val = row['pred_value']

        if pd.isna(true_val):
            continue

        pred_round = round(pred_val) if pd.notna(pred_val) else np.nan
        correct = (true_val == pred_round) if pd.notna(pred_round) else np.nan

        # Determine category from question name
        q = row['question']
        if q.startswith('TS'):
            category = 'T_Support'
        elif q.startswith('TD'):
            category = 'T_Delay'
        elif q.startswith('FS'):
            category = 'F_Support'
        elif q.startswith('FD'):
            category = 'F_Delay'
        else:
            category = 'Unknown'

        detail_rows.append({
            'row_id': row['row_id'],
            'question': q,
            'category': category,
            'true_mist': int(true_val),
            'pred_mist_mean': pred_val,
            'pred_mist_round': pred_round,
            'correct': 1 if correct else 0
        })

    df_detail = pd.DataFrame(detail_rows)

    if len(df_detail) == 0:
        return df_detail, None, None, None

    df_clean = df_detail.dropna(subset=['true_mist', 'pred_mist_round'])

    if len(df_clean) == 0:
        return df_detail, None, None, None

    # ========== Per-category metrics ==========
    category_metrics = []
    categories = ['T_Support', 'T_Delay', 'F_Support', 'F_Delay']

    for cat in categories:
        df_cat = df_clean[df_clean['category'] == cat]
        if len(df_cat) == 0:
            continue

        y_true = df_cat['true_mist'].astype(int).values
        y_pred = df_cat['pred_mist_round'].astype(int).values

        category_metrics.append({
            'Category': cat,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'N': len(y_true)
        })

    # Overall metrics
    y_true_all = df_clean['true_mist'].astype(int).values
    y_pred_all = df_clean['pred_mist_round'].astype(int).values

    category_metrics.append({
        'Category': 'OVERALL',
        'Accuracy': accuracy_score(y_true_all, y_pred_all),
        'Precision': precision_score(y_true_all, y_pred_all, zero_division=0),
        'Recall': recall_score(y_true_all, y_pred_all, zero_division=0),
        'F1': f1_score(y_true_all, y_pred_all, zero_division=0),
        'N': len(y_true_all)
    })

    df_metrics = pd.DataFrame(category_metrics)

    # ========== Confusion Matrix ==========
    cm = confusion_matrix(y_true_all, y_pred_all)

    # ========== SDT (Signal Detection Theory) ==========
    # For SDT: "signal" = true items (TS, TD), "noise" = false items (FS, FD)
    # Hit = LLM says "true" (1) for a true item where human said "true" (1)
    # False alarm = LLM says "true" (1) for a false item where human said "false" (0)
    # We compute based on LLM responses to TRUE vs FALSE statement categories
    df_true_items = df_clean[df_clean['category'].isin(['T_Support', 'T_Delay'])]
    df_false_items = df_clean[df_clean['category'].isin(['F_Support', 'F_Delay'])]

    sdt_metrics = {}

    if len(df_true_items) > 0 and len(df_false_items) > 0:
        # Hit rate: proportion of times LLM said 1 for true items
        hit_rate = df_true_items['pred_mist_round'].mean()
        # False alarm rate: proportion of times LLM said 1 for false items
        false_alarm_rate = df_false_items['pred_mist_round'].mean()

        # Correct for extreme values (0 or 1) to avoid infinite z-scores
        n_true = len(df_true_items)
        n_false = len(df_false_items)
        hit_rate_adj = np.clip(hit_rate, 0.5 / n_true, 1 - 0.5 / n_true)
        fa_rate_adj = np.clip(false_alarm_rate, 0.5 / n_false, 1 - 0.5 / n_false)

        z_hit = norm.ppf(hit_rate_adj)
        z_fa = norm.ppf(fa_rate_adj)

        d_prime = z_hit - z_fa
        c_bias = -0.5 * (z_hit + z_fa)

        sdt_metrics = {
            'Hit_Rate': hit_rate,
            'False_Alarm_Rate': false_alarm_rate,
            'Hit_Rate_Adj': hit_rate_adj,
            'FA_Rate_Adj': fa_rate_adj,
            'd_prime': d_prime,
            'c_bias': c_bias,
            'N_True_Items': n_true,
            'N_False_Items': n_false
        }

    overall_row = df_metrics[df_metrics['Category'] == 'OVERALL']
    if len(overall_row) > 0:
        m = overall_row.iloc[0]
        print(f"    Accuracy: {m['Accuracy']:.4f}, F1: {m['F1']:.4f}, N: {m['N']}")
    if sdt_metrics:
        print(f"    d': {sdt_metrics['d_prime']:.4f}, c: {sdt_metrics['c_bias']:.4f}")

    return df_detail, df_metrics, cm, sdt_metrics


def evaluate_affect_ternary(df_agg, task_name, questions):
    """
    Collapse continuous affect scores (0-100) into 3 classes and evaluate:
      0 = negative (0-33), 1 = neutral (34-66), 2 = positive (67-100)
    Returns: ternary_metrics dict, 3x3 confusion matrix.
    """
    df_task = df_agg[df_agg['task'] == task_name].copy()
    df_task = df_task.dropna(subset=['true_value', 'pred_value'])

    if len(df_task) == 0:
        return None, None

    y_true = df_task['true_value'].apply(discretize_affect_ternary).values
    y_pred = df_task['pred_value'].apply(discretize_affect_ternary).values

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    if len(y_true) == 0:
        return None, None

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'N': len(y_true)
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"    Ternary: Accuracy={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, N={metrics['N']}")
    return metrics, cm


def compute_distribution(df_agg, task_name, scale='affect'):
    """
    Count true vs predicted value distributions.
    - scale='affect': bins 0-100 into deciles (0-9, 10-19, ..., 90-100)
    - scale='ccb': counts each integer level (1-5)
    Returns: DataFrame with columns [bin, true_count, pred_count]
    """
    df_task = df_agg[df_agg['task'] == task_name].copy()
    df_task = df_task.dropna(subset=['true_value'])

    if len(df_task) == 0:
        return None

    if scale == 'affect':
        bins = list(range(0, 101, 10))
        labels = [f'{b}-{b+9}' for b in range(0, 100, 10)]

        true_series = pd.cut(df_task['true_value'], bins=bins, labels=labels,
                             include_lowest=True, right=False)
        pred_series = pd.cut(df_task['pred_value'].dropna(), bins=bins, labels=labels,
                             include_lowest=True, right=False)

        true_counts = true_series.value_counts().sort_index()
        pred_counts = pd.cut(df_task['pred_value'].dropna(), bins=bins, labels=labels,
                             include_lowest=True, right=False).value_counts().sort_index()

        dist_rows = []
        for label in labels:
            dist_rows.append({
                'bin': label,
                'true_count': int(true_counts.get(label, 0)),
                'pred_count': int(pred_counts.get(label, 0))
            })
    else:  # ccb: integer levels 1-5
        dist_rows = []
        for level in [1, 2, 3, 4, 5]:
            true_count = int((df_task['true_value'].round() == level).sum())
            pred_count = int((df_task['pred_value'].dropna().round() == level).sum())
            dist_rows.append({
                'bin': str(level),
                'true_count': true_count,
                'pred_count': pred_count
            })

    return pd.DataFrame(dist_rows)


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
