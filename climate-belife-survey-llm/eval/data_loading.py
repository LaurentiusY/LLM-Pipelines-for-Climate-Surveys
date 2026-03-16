import json
import numpy as np
import pandas as pd
from shared.io_utils import read_csv_robust
from eval.utils import convert_share_true, convert_share_response, calc_true_trees


def load_data(input_csv, input_results_json):
    print("[1] Loading data...")

    df_original = read_csv_robust(input_csv)
    df_original.insert(0, 'row_id', range(1, len(df_original) + 1))
    print(f"    Original data: {len(df_original)} rows")

    with open(input_results_json, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    print(f"    LLM results: {len(results_data['results'])} records")

    metadata = results_data.get('metadata', {})
    print(f"    Model: {metadata.get('model', 'unknown')}")
    print(f"    Method: {metadata.get('method', metadata.get('technique', 'unknown'))}")
    print(f"    Repeat times: {metadata.get('repeat_times', 'unknown')}")
    print(f"    LLM seed: {metadata.get('llm_random_seed', 'not set')}")

    return df_original, results_data


def extract_raw_data(results_data, df_original, belief_cols, policy_cols, share_col, wept_cols):
    """
    提取原始数据（每人每题每次的回答）
    """
    print("[2] Extracting raw data (per person per question per repeat)...")

    true_values = {}

    for _, row in df_original.iterrows():
        row_id = row['row_id']

        for q, col in belief_cols.items():
            if col in row.index:
                true_values[(row_id, 'belief', q)] = row[col]

        for q, col in policy_cols.items():
            if col in row.index:
                true_values[(row_id, 'policy', q)] = row[col]

        if share_col in row.index:
            true_values[(row_id, 'share', 'Q1')] = convert_share_true(row[share_col])

        true_values[(row_id, 'trees', 'total')] = calc_true_trees(row, wept_cols)

    raw_records = []

    for result in results_data['results']:
        row_id = result['row_id']
        repeat = result.get('repeat', 1)

        # ========== Belief (Q1-Q4) ==========
        belief_data = result.get('belief')
        for i in range(1, 5):
            q = f"Q{i}"

            pred_val = None
            if isinstance(belief_data, dict):
                pred_val = belief_data.get(q)

            if isinstance(pred_val, (int, float)):
                pred_val = float(pred_val)
            else:
                try:
                    pred_val = float(pred_val)
                except:
                    pred_val = np.nan

            true_val = true_values.get((row_id, 'belief', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'belief',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': pred_val - true_val if pd.notna(pred_val) and pd.notna(true_val) else np.nan
            })

        # ========== Policy (Q1-Q9) ==========
        policy_data = result.get('policy')
        for i in range(1, 10):
            q = f"Q{i}"

            pred_val = None
            if isinstance(policy_data, dict):
                pred_val = policy_data.get(q)

            if pred_val == "NA" or pred_val is None:
                pred_val = np.nan
            elif isinstance(pred_val, (int, float)):
                pred_val = float(pred_val)
            else:
                try:
                    pred_val = float(pred_val)
                except:
                    pred_val = np.nan

            true_val = true_values.get((row_id, 'policy', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'policy',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': pred_val - true_val if pd.notna(pred_val) and pd.notna(true_val) else np.nan
            })

        # ========== Share ==========
        pred_share = convert_share_response(result.get('share'))
        true_share = true_values.get((row_id, 'share', 'Q1'), np.nan)

        raw_records.append({
            'row_id': row_id,
            'repeat': repeat,
            'task': 'share',
            'question': 'Q1',
            'true_value': true_share,
            'pred_value': pred_share,
            'error': np.nan
        })

        # ========== Trees ==========
        pred_trees = result.get('trees')
        if isinstance(pred_trees, (int, float)):
            pred_trees = float(pred_trees)
        else:
            try:
                pred_trees = float(pred_trees)
            except:
                pred_trees = np.nan

        true_trees = true_values.get((row_id, 'trees', 'total'), np.nan)

        raw_records.append({
            'row_id': row_id,
            'repeat': repeat,
            'task': 'trees',
            'question': 'total',
            'true_value': true_trees,
            'pred_value': pred_trees,
            'error': pred_trees - true_trees if pd.notna(pred_trees) and pd.notna(true_trees) else np.nan
        })

    df_raw = pd.DataFrame(raw_records)
    print(f"    Raw records: {len(df_raw)} rows")

    return df_raw


def extract_plans_cot(results_data):
    """
    把 CoT Stage1 plan 拉平成表（evidence/key_factors/response_guideline）
    """
    rows = []
    for r in results_data.get('results', []):
        row_id = r.get('row_id')
        repeat = r.get('repeat', 1)
        plans = r.get('plans', {})

        for task in ['belief', 'policy', 'share', 'trees']:
            p = plans.get(task)
            if p is None:
                continue

            evidence = None
            key_factors = None
            response_guideline = None

            if isinstance(p, dict):
                evidence = p.get('evidence')
                key_factors = p.get('key_factors')
                response_guideline = p.get('response_guideline')

            evidence_n = len(evidence) if isinstance(evidence, list) else np.nan
            key_factors_len = len(key_factors) if isinstance(key_factors, str) else np.nan
            response_guideline_len = len(response_guideline) if isinstance(response_guideline, str) else np.nan

            rows.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': task,
                'plan_raw': json.dumps(p, ensure_ascii=False) if not isinstance(p, str) else p,
                'evidence_n': evidence_n,
                'key_factors_len': key_factors_len,
                'response_guideline_len': response_guideline_len
            })

    return pd.DataFrame(rows)


def extract_plans_vbn(results_data):
    """
    把 VBN Stage1 plan 拉平成表（evidence/VBN AC,AR,PN/synthesis）
    """
    rows = []
    for r in results_data.get('results', []):
        row_id = r.get('row_id')
        repeat = r.get('repeat', 1)
        plans = r.get('plans', {})

        for task in ['belief', 'policy', 'share', 'trees']:
            p = plans.get(task)
            if p is None:
                continue

            evidence = None
            synthesis = None
            vbn = None

            if isinstance(p, dict):
                evidence = p.get('evidence')
                synthesis = p.get('synthesis')
                vbn = p.get('VBN') or {}

            evidence_n = len(evidence) if isinstance(evidence, list) else np.nan
            synthesis_len = len(synthesis) if isinstance(synthesis, str) else np.nan

            AC = vbn.get('AC') if isinstance(vbn, dict) else None
            AR = vbn.get('AR') if isinstance(vbn, dict) else None
            PN = vbn.get('PN') if isinstance(vbn, dict) else None

            plan_raw = json.dumps(p, ensure_ascii=False) if not isinstance(p, str) else p

            rows.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': task,
                'AC': AC,
                'AR': AR,
                'PN': PN,
                'evidence_n': evidence_n,
                'synthesis_len': synthesis_len,
                'plan_raw': plan_raw
            })

    return pd.DataFrame(rows)


def aggregate_data(df_raw):
    """
    按 row_id + task + question 聚合（每人N次重复的均值）
    """
    print("[3] Aggregating data (mean of repeats per person)...")

    df_agg = df_raw.groupby(['row_id', 'task', 'question']).agg({
        'true_value': 'first',
        'pred_value': 'mean',
        'error': 'mean'
    }).reset_index()

    print(f"    Aggregated records: {len(df_agg)} rows")

    return df_agg
