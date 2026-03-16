import json
import numpy as np
import pandas as pd
from shared.io_utils import read_csv_robust
from eval.utils import convert_mist_response


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


def extract_raw_data(results_data, df_original,
                     affect_science_cols, affect_action_cols,
                     ccb_cols, mist_cols):
    """
    Extract raw data (per person per question per repeat) for all 4 tasks.
    """
    print("[2] Extracting raw data (per person per question per repeat)...")

    true_values = {}

    for _, row in df_original.iterrows():
        row_id = row['row_id']

        # Affect Science (S1-S10)
        for q, col in affect_science_cols.items():
            if col in row.index:
                true_values[(row_id, 'affect_science', q)] = row[col]

        # Affect Action (A1-A10)
        for q, col in affect_action_cols.items():
            if col in row.index:
                true_values[(row_id, 'affect_action', q)] = row[col]

        # CCB (Q1-Q3)
        for q, col in ccb_cols.items():
            if col in row.index:
                true_values[(row_id, 'ccb', q)] = row[col]

        # MIST (TS1-5, TD1-5, FS1-5, FD1-5)
        for q, col in mist_cols.items():
            if col in row.index:
                true_values[(row_id, 'mist', q)] = row[col]

    raw_records = []

    for result in results_data['results']:
        row_id = result['row_id']
        repeat = result.get('repeat', 1)

        # ========== Affect Science (S1-S10) ==========
        affect_sci_data = result.get('affect_science')
        for i in range(1, 11):
            q = f"S{i}"

            pred_val = None
            if isinstance(affect_sci_data, dict):
                pred_val = affect_sci_data.get(q)

            if isinstance(pred_val, (int, float)):
                pred_val = float(pred_val)
            else:
                try:
                    pred_val = float(pred_val)
                except:
                    pred_val = np.nan

            true_val = true_values.get((row_id, 'affect_science', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'affect_science',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': pred_val - true_val if pd.notna(pred_val) and pd.notna(true_val) else np.nan
            })

        # ========== Affect Action (A1-A10) ==========
        affect_act_data = result.get('affect_action')
        for i in range(1, 11):
            q = f"A{i}"

            pred_val = None
            if isinstance(affect_act_data, dict):
                pred_val = affect_act_data.get(q)

            if isinstance(pred_val, (int, float)):
                pred_val = float(pred_val)
            else:
                try:
                    pred_val = float(pred_val)
                except:
                    pred_val = np.nan

            true_val = true_values.get((row_id, 'affect_action', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'affect_action',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': pred_val - true_val if pd.notna(pred_val) and pd.notna(true_val) else np.nan
            })

        # ========== CCB (Q1-Q3) ==========
        ccb_data = result.get('ccb')
        for i in range(1, 4):
            q = f"Q{i}"

            pred_val = None
            if isinstance(ccb_data, dict):
                pred_val = ccb_data.get(q)

            if pred_val == "NA" or pred_val is None:
                pred_val = np.nan
            elif isinstance(pred_val, (int, float)):
                pred_val = float(pred_val)
            else:
                try:
                    pred_val = float(pred_val)
                except:
                    pred_val = np.nan

            true_val = true_values.get((row_id, 'ccb', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'ccb',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': pred_val - true_val if pd.notna(pred_val) and pd.notna(true_val) else np.nan
            })

        # ========== MIST (20 binary items) ==========
        mist_data = result.get('mist')

        # True Support items (TS1-TS5)
        for i in range(1, 6):
            q = f"TS{i}"
            pred_val = np.nan
            if isinstance(mist_data, dict):
                pred_val = convert_mist_response(mist_data.get(q))
            true_val = true_values.get((row_id, 'mist', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'mist',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': np.nan
            })

        # True Delay items (TD1-TD5)
        for i in range(1, 6):
            q = f"TD{i}"
            pred_val = np.nan
            if isinstance(mist_data, dict):
                pred_val = convert_mist_response(mist_data.get(q))
            true_val = true_values.get((row_id, 'mist', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'mist',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': np.nan
            })

        # False Support items (FS1-FS5)
        for i in range(1, 6):
            q = f"FS{i}"
            pred_val = np.nan
            if isinstance(mist_data, dict):
                pred_val = convert_mist_response(mist_data.get(q))
            true_val = true_values.get((row_id, 'mist', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'mist',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': np.nan
            })

        # False Delay items (FD1-FD5)
        for i in range(1, 6):
            q = f"FD{i}"
            pred_val = np.nan
            if isinstance(mist_data, dict):
                pred_val = convert_mist_response(mist_data.get(q))
            true_val = true_values.get((row_id, 'mist', q), np.nan)

            raw_records.append({
                'row_id': row_id,
                'repeat': repeat,
                'task': 'mist',
                'question': q,
                'true_value': true_val,
                'pred_value': pred_val,
                'error': np.nan
            })

    df_raw = pd.DataFrame(raw_records)
    print(f"    Raw records: {len(df_raw)} rows")

    return df_raw


def extract_affect_delta(results_data, df_original,
                         affect_science_cols, affect_action_cols):
    """
    Compute affect delta = Affect_after - Affect_T1_1 (baseline) for each person/tweet.
    Returns a DataFrame with columns:
      row_id, task, question, baseline, true_after, true_delta, pred_after, pred_delta
    """
    print("[2b] Extracting affect delta (change from baseline)...")

    # Build lookup: row_id -> baseline affect
    baseline_map = {}
    for _, row in df_original.iterrows():
        rid = row['row_id']
        if 'Affect_T1_1' in row.index and pd.notna(row['Affect_T1_1']):
            baseline_map[rid] = float(row['Affect_T1_1'])

    # Build lookup: (row_id, task, question) -> true after value
    true_values = {}
    for _, row in df_original.iterrows():
        rid = row['row_id']
        for q, col in affect_science_cols.items():
            if col in row.index:
                true_values[(rid, 'affect_science', q)] = row[col]
        for q, col in affect_action_cols.items():
            if col in row.index:
                true_values[(rid, 'affect_action', q)] = row[col]

    delta_records = []

    for result in results_data['results']:
        row_id = result['row_id']
        baseline = baseline_map.get(row_id, np.nan)

        for task_name, data_key, q_prefix, n_items in [
            ('affect_science', 'affect_science', 'S', 10),
            ('affect_action', 'affect_action', 'A', 10),
        ]:
            task_data = result.get(data_key)
            for i in range(1, n_items + 1):
                q = f'{q_prefix}{i}'

                pred_after = np.nan
                if isinstance(task_data, dict):
                    raw = task_data.get(q)
                    try:
                        pred_after = float(raw)
                    except (TypeError, ValueError):
                        pass

                true_after = true_values.get((row_id, task_name, q), np.nan)

                true_delta = true_after - baseline if pd.notna(true_after) and pd.notna(baseline) else np.nan
                pred_delta = pred_after - baseline if pd.notna(pred_after) and pd.notna(baseline) else np.nan

                delta_records.append({
                    'row_id': row_id,
                    'task': task_name,
                    'question': q,
                    'baseline': baseline,
                    'true_after': true_after,
                    'true_delta': true_delta,
                    'pred_after': pred_after,
                    'pred_delta': pred_delta,
                    'delta_error': pred_delta - true_delta if pd.notna(pred_delta) and pd.notna(true_delta) else np.nan
                })

    df_delta = pd.DataFrame(delta_records)
    print(f"    Delta records: {len(df_delta)} rows ({df_delta['baseline'].notna().sum()} with valid baseline)")
    return df_delta


def extract_plans_cot(results_data):
    """
    Extract CoT Stage1 plans (evidence/key_factors/response_guideline)
    """
    rows = []
    for r in results_data.get('results', []):
        row_id = r.get('row_id')
        repeat = r.get('repeat', 1)
        plans = r.get('plans', {})

        for task in ['affect_science', 'affect_action', 'ccb', 'mist']:
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
    Extract VBN Stage1 plans (evidence/VBN AC,AR,PN/synthesis)
    """
    rows = []
    for r in results_data.get('results', []):
        row_id = r.get('row_id')
        repeat = r.get('repeat', 1)
        plans = r.get('plans', {})

        for task in ['affect_science', 'affect_action', 'ccb', 'mist']:
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
    Aggregate by row_id + task + question (mean of N repeats per person)
    """
    print("[3] Aggregating data (mean of repeats per person)...")

    df_agg = df_raw.groupby(['row_id', 'task', 'question']).agg({
        'true_value': 'first',
        'pred_value': 'mean',
        'error': 'mean'
    }).reset_index()

    print(f"    Aggregated records: {len(df_agg)} rows")

    return df_agg
