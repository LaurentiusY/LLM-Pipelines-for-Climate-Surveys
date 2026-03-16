import os
import sys
from eval.data_loading import load_data, extract_raw_data, extract_plans_vbn, aggregate_data
from eval.metrics import evaluate_continuous, evaluate_share, evaluate_trees, evaluate_correlation
from eval.group_analysis import run_group_analysis
from eval.excel_output import (
    save_plans_to_excel_vbn, save_raw_to_excel,
    save_to_excel, save_group_analysis_to_excel,
)

# =========================================================================
# 【配置区】— 列名映射（与方法无关，数据集决定）
# =========================================================================
INPUT_ORIGINAL_CSV = 'data/climate_belief_data_notimers.csv'

GROUP_COLS = {
    'country': 'country',
}

BELIEF_COLS = {
    'Q1': 'Belief.in.CC_4',
    'Q2': 'Belief.in.CC_1',
    'Q3': 'Belief.in.CC_2',
    'Q4': 'Belief.in.CC_5'
}

POLICY_COLS = {
    'Q1': 'CC_policy_2',
    'Q2': 'CC_policy_7',
    'Q3': 'CC_policy_3',
    'Q4': 'CC_policy_8',
    'Q5': 'CC_policy_6',
    'Q6': 'CC_policy_10',
    'Q7': 'CC_policy_1',
    'Q8': 'CC_policy_5',
    'Q9': 'CC_policy_9'
}

SHARE_COL = 'Share'

WEPT_COLS = {
    'Q2': 'WEPT1confirm',
    'Q3': 'WEPT2confirm',
    'Q4': 'WEPT3confirm',
    'Q5': 'WEPT4confirm',
    'Q6': 'WEPT5confirm',
    'Q7': 'WEPT6confirm',
    'Q8': 'WEPT7confirm',
    'Q9': 'WEPT8confirm'
}


def main(run_dir):
    input_results = os.path.join(run_dir, 'results.json')
    output_xlsx = os.path.join(run_dir, 'evaluation.xlsx')
    output_raw = os.path.join(run_dir, 'evaluation_raw.xlsx')
    output_group = os.path.join(run_dir, 'evaluation_by_group.xlsx')
    output_plans = os.path.join(run_dir, 'evaluation_plans.xlsx')

    print("=" * 60)
    print("Evaluation Script (With Group Analysis)")
    print("=" * 60)

    df_original, results_data = load_data(INPUT_ORIGINAL_CSV, input_results)
    df_plans = extract_plans_vbn(results_data)
    save_plans_to_excel_vbn(df_plans, output_plans)

    df_raw = extract_raw_data(results_data, df_original, BELIEF_COLS, POLICY_COLS, SHARE_COL, WEPT_COLS)
    df_agg = aggregate_data(df_raw)

    belief_detail, belief_metrics = evaluate_continuous(
        df_raw, df_agg, 'belief', [f'Q{i}' for i in range(1, 5)]
    )
    policy_detail, policy_metrics = evaluate_continuous(
        df_raw, df_agg, 'policy', [f'Q{i}' for i in range(1, 10)]
    )
    share_detail, share_metrics, share_cm = evaluate_share(df_raw, df_agg)
    trees_detail, trees_results = evaluate_trees(df_raw, df_agg)

    print("\n[Evaluating correlations]")
    print("  -> Correlation (Belief):")
    belief_correlation = evaluate_correlation(df_agg, 'belief', [f'Q{i}' for i in range(1, 5)])
    print("  -> Correlation (Policy):")
    policy_correlation = evaluate_correlation(df_agg, 'policy', [f'Q{i}' for i in range(1, 10)])

    save_raw_to_excel(df_raw, output_raw)
    save_to_excel(belief_detail, belief_metrics,
                  policy_detail, policy_metrics,
                  share_detail, share_metrics, share_cm,
                  trees_detail, trees_results,
                  output_xlsx,
                  belief_correlation=belief_correlation,
                  policy_correlation=policy_correlation)

    valid_group_cols = {}
    for key, col in GROUP_COLS.items():
        if col in df_original.columns:
            valid_group_cols[key] = col
        else:
            print(f"    Warning: Group column '{col}' not found in data, skipping...")

    if valid_group_cols:
        group_results = run_group_analysis(df_agg, df_original, valid_group_cols)
        save_group_analysis_to_excel(group_results, output_group)
    else:
        print("\n[Warning] No valid group columns found. Skipping group analysis.")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if belief_metrics is not None and len(belief_metrics) > 0:
        overall = belief_metrics[(belief_metrics['scale'] == 'original') &
                                 (belief_metrics['question'] == 'OVERALL')]
        if len(overall) > 0:
            m = overall.iloc[0]
            print(f"\nBelief: Mean_Error={m['Mean_Error']:.2f}, MAE={m['MAE']:.2f}, N={m['N']}")

    if policy_metrics is not None and len(policy_metrics) > 0:
        overall = policy_metrics[(policy_metrics['scale'] == 'original') &
                                 (policy_metrics['question'] == 'OVERALL')]
        if len(overall) > 0:
            m = overall.iloc[0]
            print(f"Policy: Mean_Error={m['Mean_Error']:.2f}, MAE={m['MAE']:.2f}, N={m['N']}")

    if share_metrics:
        print(f"Share: Accuracy={share_metrics['Accuracy']:.4f}, F1={share_metrics['F1']:.4f}, N={share_metrics['N']}")

    if trees_results:
        if 'continuous' in trees_results:
            m = trees_results['continuous']
            print(f"Trees (continuous): Mean_Error={m['Mean_Error']:.2f}, MAE={m['MAE']:.2f}, N={m['N']}")
        if 'binary' in trees_results:
            m = trees_results['binary']
            print(f"Trees (binary): Accuracy={m['Accuracy']:.4f}, F1={m['F1']:.4f}")
        if 'ternary' in trees_results:
            m = trees_results['ternary']
            print(f"Trees (ternary): Accuracy={m['Accuracy']:.4f}, F1={m['F1']:.4f}")

    print(f"\n[DONE] All results saved to: {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.run_vbn <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
