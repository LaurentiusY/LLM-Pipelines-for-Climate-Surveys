import os
import sys
from eval.data_loading import load_data, extract_raw_data, extract_plans_vbn, aggregate_data, extract_affect_delta
from eval.metrics import (
    evaluate_continuous, evaluate_ccb, evaluate_mist,
    evaluate_affect_ternary, compute_distribution, evaluate_correlation
)
from eval.group_analysis import run_group_analysis
from eval.excel_output import (
    save_plans_to_excel_vbn, save_raw_to_excel,
    save_to_excel, save_group_analysis_to_excel,
)

# =========================================================================
# Column mappings (dataset-specific, method-independent)
# =========================================================================
INPUT_ORIGINAL_CSV = 'data/Showdown_short.csv'

GROUP_COLS = {
    'Country': 'Country',
    'Inoculation_Group': 'Inoculation_Group',
}

# Affect Science: 10 items (S1-S10), 0-100 continuous
AFFECT_SCIENCE_COLS = {
    'S1': 'Sci_1_Affect_1',
    'S2': 'Sci_2_Affect_1',
    'S3': 'Sci_3_Affect_1',
    'S4': 'Sci_4_Affect_1',
    'S5': 'Sci_5_Affect_1',
    'S6': 'Sci_6_Affect_1',
    'S7': 'Sci_7_Affect_1',
    'S8': 'Sci_8_Affect_1',
    'S9': 'Sci_9_Affect_1',
    'S10': 'Sci_10_Affect_1',
}

# Affect Action: 10 items (A1-A10), 0-100 continuous
AFFECT_ACTION_COLS = {
    'A1': 'Act_1_Affect_1',
    'A2': 'Act_2_Affect_1',
    'A3': 'Act_3_Affect_1',
    'A4': 'Act_4_Affect_1',
    'A5': 'Act_5_Affect_1',
    'A6': 'Act_6_Affect_1',
    'A7': 'Act_7_Affect_1',
    'A8': 'Act_8_Affect_1',
    'A9': 'Act_9_Affect_1',
    'A10': 'Act_10_Affect_1',
}

# CCB: 3 items (Q1-Q3), 1-5 continuous
CCB_COLS = {
    'Q1': 'CCB_real',
    'Q2': 'CCB_cause',
    'Q3': 'CCB_cons',
}

# MIST: 20 binary items
MIST_COLS = {
    'TS1': 'T_Support_1',
    'TS2': 'T_Support_2',
    'TS3': 'T_Support_3',
    'TS4': 'T_Support_4',
    'TS5': 'T_Support_5',
    'TD1': 'T_Delay_1',
    'TD2': 'T_Delay_2',
    'TD3': 'T_Delay_3',
    'TD4': 'T_Delay_4',
    'TD5': 'T_Delay_5',
    'FS1': 'F_Support_1',
    'FS2': 'F_Support_2',
    'FS3': 'F_Support_3',
    'FS4': 'F_Support_4',
    'FS5': 'F_Support_5',
    'FD1': 'F_Delay_1',
    'FD2': 'F_Delay_2',
    'FD3': 'F_Delay_3',
    'FD4': 'F_Delay_4',
    'FD5': 'F_Delay_5',
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

    df_raw = extract_raw_data(results_data, df_original,
                              AFFECT_SCIENCE_COLS, AFFECT_ACTION_COLS,
                              CCB_COLS, MIST_COLS)
    df_agg = aggregate_data(df_raw)

    # Evaluate Affect Science (0-100 continuous)
    affect_sci_detail, affect_sci_metrics = evaluate_continuous(
        df_raw, df_agg, 'affect_science', [f'S{i}' for i in range(1, 11)]
    )

    # Evaluate Affect Action (0-100 continuous)
    affect_act_detail, affect_act_metrics = evaluate_continuous(
        df_raw, df_agg, 'affect_action', [f'A{i}' for i in range(1, 11)]
    )

    # Evaluate CCB (1-5 continuous)
    ccb_detail, ccb_metrics = evaluate_ccb(df_raw, df_agg)

    # Evaluate MIST (binary)
    mist_detail, mist_metrics, mist_cm, mist_sdt = evaluate_mist(df_raw, df_agg)

    # ===== New statistics =====
    print("\n[Evaluating additional statistics]")

    print("  -> Ternary (Affect Science):")
    affect_sci_ternary_metrics, affect_sci_ternary_cm = evaluate_affect_ternary(
        df_agg, 'affect_science', [f'S{i}' for i in range(1, 11)])
    print("  -> Ternary (Affect Action):")
    affect_act_ternary_metrics, affect_act_ternary_cm = evaluate_affect_ternary(
        df_agg, 'affect_action', [f'A{i}' for i in range(1, 11)])

    print("  -> Distribution (Affect Science, Affect Action, CCB):")
    affect_sci_distribution = compute_distribution(df_agg, 'affect_science', scale='affect')
    affect_act_distribution = compute_distribution(df_agg, 'affect_action', scale='affect')
    ccb_distribution = compute_distribution(df_agg, 'ccb', scale='ccb')

    print("  -> Correlation (Affect Science):")
    affect_sci_correlation = evaluate_correlation(
        df_agg, 'affect_science', [f'S{i}' for i in range(1, 11)])
    print("  -> Correlation (Affect Action):")
    affect_act_correlation = evaluate_correlation(
        df_agg, 'affect_action', [f'A{i}' for i in range(1, 11)])
    print("  -> Correlation (CCB):")
    ccb_correlation = evaluate_correlation(df_agg, 'ccb', ['Q1', 'Q2', 'Q3'])

    df_delta = extract_affect_delta(results_data, df_original,
                                    AFFECT_SCIENCE_COLS, AFFECT_ACTION_COLS)
    affect_sci_delta = df_delta[df_delta['task'] == 'affect_science'].copy()
    affect_act_delta = df_delta[df_delta['task'] == 'affect_action'].copy()

    save_raw_to_excel(df_raw, output_raw)
    save_to_excel(affect_sci_detail, affect_sci_metrics,
                  affect_act_detail, affect_act_metrics,
                  ccb_detail, ccb_metrics,
                  mist_detail, mist_metrics, mist_cm, mist_sdt,
                  output_xlsx,
                  affect_sci_ternary_metrics=affect_sci_ternary_metrics,
                  affect_sci_ternary_cm=affect_sci_ternary_cm,
                  affect_act_ternary_metrics=affect_act_ternary_metrics,
                  affect_act_ternary_cm=affect_act_ternary_cm,
                  affect_sci_distribution=affect_sci_distribution,
                  affect_act_distribution=affect_act_distribution,
                  ccb_distribution=ccb_distribution,
                  affect_sci_correlation=affect_sci_correlation,
                  affect_act_correlation=affect_act_correlation,
                  ccb_correlation=ccb_correlation,
                  affect_sci_delta=affect_sci_delta,
                  affect_act_delta=affect_act_delta)

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

    if affect_sci_metrics is not None and len(affect_sci_metrics) > 0:
        overall = affect_sci_metrics[(affect_sci_metrics['scale'] == 'original') &
                                     (affect_sci_metrics['question'] == 'OVERALL')]
        if len(overall) > 0:
            m = overall.iloc[0]
            print(f"\nAffect Science: Mean_Error={m['Mean_Error']:.2f}, MAE={m['MAE']:.2f}, N={m['N']}")

    if affect_act_metrics is not None and len(affect_act_metrics) > 0:
        overall = affect_act_metrics[(affect_act_metrics['scale'] == 'original') &
                                     (affect_act_metrics['question'] == 'OVERALL')]
        if len(overall) > 0:
            m = overall.iloc[0]
            print(f"Affect Action: Mean_Error={m['Mean_Error']:.2f}, MAE={m['MAE']:.2f}, N={m['N']}")

    if ccb_metrics is not None and len(ccb_metrics) > 0:
        overall = ccb_metrics[(ccb_metrics['scale'] == 'original') &
                              (ccb_metrics['question'] == 'OVERALL')]
        if len(overall) > 0:
            m = overall.iloc[0]
            print(f"CCB: Mean_Error={m['Mean_Error']:.2f}, MAE={m['MAE']:.2f}, N={m['N']}")

    if mist_metrics is not None and len(mist_metrics) > 0:
        overall = mist_metrics[mist_metrics['Category'] == 'OVERALL']
        if len(overall) > 0:
            m = overall.iloc[0]
            print(f"MIST: Accuracy={m['Accuracy']:.4f}, F1={m['F1']:.4f}, N={m['N']}")

    if mist_sdt:
        print(f"MIST SDT: d'={mist_sdt['d_prime']:.4f}, c={mist_sdt['c_bias']:.4f}")

    print(f"\n[DONE] All results saved to: {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.run_vbn <run_dir>")
        sys.exit(1)
    main(sys.argv[1])
