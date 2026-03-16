import pandas as pd


def save_plans_to_excel_cot(df_plans, output_path):
    print(f"\n[Saving stage1 plans to Excel: {output_path}]")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_plans.to_excel(writer, sheet_name='Plans_Raw', index=False)

        if len(df_plans) > 0:
            summary = df_plans.groupby(['task']).agg({
                'evidence_n': ['count', 'mean', 'std'],
                'key_factors_len': ['mean', 'std'],
                'response_guideline_len': ['mean', 'std']
            }).round(2)
            summary.columns = ['_'.join(c) for c in summary.columns]
            summary = summary.reset_index()
            summary.to_excel(writer, sheet_name='Plans_Summary', index=False)

    print("    Saved successfully!")


def save_plans_to_excel_vbn(df_plans, output_path):
    print(f"\n[Saving stage1 plans to Excel: {output_path}]")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_plans.to_excel(writer, sheet_name='Plans_Raw', index=False)

        if len(df_plans) > 0:
            label_dist = (
                df_plans.groupby(['task', 'AC', 'AR', 'PN'])
                .size()
                .reset_index(name='count')
                .sort_values(['task', 'count'], ascending=[True, False])
            )
            label_dist.to_excel(writer, sheet_name='Plans_VBN_Distribution', index=False)

            summary = df_plans.groupby(['task']).agg({
                'evidence_n': ['count', 'mean', 'std'],
                'synthesis_len': ['mean', 'std']
            }).round(2)
            summary.columns = ['_'.join(c) for c in summary.columns]
            summary = summary.reset_index()
            summary.to_excel(writer, sheet_name='Plans_Summary', index=False)

    print("    Saved successfully!")


def save_raw_to_excel(df_raw, output_path):
    """Save raw data (per person per question per repeat) to Excel"""
    print(f"\n[Saving raw data to Excel: {output_path}]")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for task in df_raw['task'].unique():
            df_task = df_raw[df_raw['task'] == task].copy()
            df_task = df_task.sort_values(['row_id', 'repeat', 'question'])

            sheet_name = f'Raw_{task}'
            # Truncate sheet name to 31 chars (Excel limit)
            sheet_name = sheet_name[:31]
            df_task.to_excel(writer, sheet_name=sheet_name, index=False)

        summary = df_raw.groupby(['task', 'question']).agg({
            'true_value': ['count', 'mean', 'std'],
            'pred_value': ['count', 'mean', 'std'],
            'error': ['mean', 'std']
        }).round(2)
        summary.columns = ['_'.join(col) for col in summary.columns]
        summary = summary.reset_index()
        summary.to_excel(writer, sheet_name='Raw_Summary', index=False)

    print(f"    Saved successfully!")


def save_to_excel(affect_science_detail, affect_science_metrics,
                  affect_action_detail, affect_action_metrics,
                  ccb_detail, ccb_metrics,
                  mist_detail, mist_metrics, mist_cm, mist_sdt,
                  output_path,
                  affect_sci_ternary_metrics=None, affect_sci_ternary_cm=None,
                  affect_act_ternary_metrics=None, affect_act_ternary_cm=None,
                  affect_sci_distribution=None, affect_act_distribution=None,
                  ccb_distribution=None,
                  affect_sci_correlation=None, affect_act_correlation=None,
                  ccb_correlation=None,
                  affect_sci_delta=None, affect_act_delta=None):
    """Save aggregated evaluation results to Excel"""
    print(f"\n[Saving aggregated results to Excel: {output_path}]")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Affect Science
        if affect_science_detail is not None and len(affect_science_detail) > 0:
            affect_science_detail.to_excel(writer, sheet_name='Affect_Science_Detail', index=False)

        if affect_science_metrics is not None and len(affect_science_metrics) > 0:
            affect_science_metrics.to_excel(writer, sheet_name='Affect_Science_Metrics', index=False)

        # Affect Action
        if affect_action_detail is not None and len(affect_action_detail) > 0:
            affect_action_detail.to_excel(writer, sheet_name='Affect_Action_Detail', index=False)

        if affect_action_metrics is not None and len(affect_action_metrics) > 0:
            affect_action_metrics.to_excel(writer, sheet_name='Affect_Action_Metrics', index=False)

        # CCB
        if ccb_detail is not None and len(ccb_detail) > 0:
            ccb_detail.to_excel(writer, sheet_name='CCB_Detail', index=False)

        if ccb_metrics is not None and len(ccb_metrics) > 0:
            ccb_metrics.to_excel(writer, sheet_name='CCB_Metrics', index=False)

        # MIST
        if mist_detail is not None and len(mist_detail) > 0:
            mist_detail.to_excel(writer, sheet_name='MIST_Detail', index=False)

        if mist_metrics is not None and len(mist_metrics) > 0:
            mist_metrics.to_excel(writer, sheet_name='MIST_Metrics', index=False)

        if mist_cm is not None:
            labels = ['False(0)', 'True(1)'][:mist_cm.shape[0]]
            pd.DataFrame(mist_cm,
                         index=[f'True:{l}' for l in labels],
                         columns=[f'Pred:{l}' for l in labels]).to_excel(
                writer, sheet_name='MIST_CM')

        if mist_sdt is not None and len(mist_sdt) > 0:
            pd.DataFrame([mist_sdt]).to_excel(writer, sheet_name='MIST_SDT', index=False)

        # ========== Ternary classification (Affect tasks) ==========
        if affect_sci_ternary_metrics is not None:
            pd.DataFrame([affect_sci_ternary_metrics]).to_excel(
                writer, sheet_name='AffectSci_Ternary_Metrics', index=False)
        if affect_sci_ternary_cm is not None:
            labels = ['Negative', 'Neutral', 'Positive']
            pd.DataFrame(affect_sci_ternary_cm,
                         index=[f'True:{l}' for l in labels],
                         columns=[f'Pred:{l}' for l in labels]).to_excel(
                writer, sheet_name='AffectSci_Ternary_CM')

        if affect_act_ternary_metrics is not None:
            pd.DataFrame([affect_act_ternary_metrics]).to_excel(
                writer, sheet_name='AffectAct_Ternary_Metrics', index=False)
        if affect_act_ternary_cm is not None:
            labels = ['Negative', 'Neutral', 'Positive']
            pd.DataFrame(affect_act_ternary_cm,
                         index=[f'True:{l}' for l in labels],
                         columns=[f'Pred:{l}' for l in labels]).to_excel(
                writer, sheet_name='AffectAct_Ternary_CM')

        # ========== Distribution analysis ==========
        if affect_sci_distribution is not None and len(affect_sci_distribution) > 0:
            affect_sci_distribution.to_excel(writer, sheet_name='AffectSci_Distribution', index=False)
        if affect_act_distribution is not None and len(affect_act_distribution) > 0:
            affect_act_distribution.to_excel(writer, sheet_name='AffectAct_Distribution', index=False)
        if ccb_distribution is not None and len(ccb_distribution) > 0:
            ccb_distribution.to_excel(writer, sheet_name='CCB_Distribution', index=False)

        # ========== Correlation analysis ==========
        if affect_sci_correlation is not None and len(affect_sci_correlation) > 0:
            affect_sci_correlation.to_excel(writer, sheet_name='AffectSci_Correlation', index=False)
        if affect_act_correlation is not None and len(affect_act_correlation) > 0:
            affect_act_correlation.to_excel(writer, sheet_name='AffectAct_Correlation', index=False)
        if ccb_correlation is not None and len(ccb_correlation) > 0:
            ccb_correlation.to_excel(writer, sheet_name='CCB_Correlation', index=False)

        # ========== Affect delta ==========
        if affect_sci_delta is not None and len(affect_sci_delta) > 0:
            affect_sci_delta.to_excel(writer, sheet_name='AffectSci_Delta', index=False)
        if affect_act_delta is not None and len(affect_act_delta) > 0:
            affect_act_delta.to_excel(writer, sheet_name='AffectAct_Delta', index=False)

        # ========== Summary ==========
        summary_rows = []

        if affect_science_metrics is not None and len(affect_science_metrics) > 0:
            for scale in ['original', 'decimal', 'discrete']:
                overall = affect_science_metrics[
                    (affect_science_metrics['scale'] == scale) &
                    (affect_science_metrics['question'] == 'OVERALL')]
                if len(overall) > 0:
                    m = overall.iloc[0]
                    summary_rows.append({
                        'Task': 'Affect_Science',
                        'Scale': scale,
                        'Mean_Error': m['Mean_Error'],
                        'MAE': m['MAE'],
                        'N': m['N']
                    })

        if affect_action_metrics is not None and len(affect_action_metrics) > 0:
            for scale in ['original', 'decimal', 'discrete']:
                overall = affect_action_metrics[
                    (affect_action_metrics['scale'] == scale) &
                    (affect_action_metrics['question'] == 'OVERALL')]
                if len(overall) > 0:
                    m = overall.iloc[0]
                    summary_rows.append({
                        'Task': 'Affect_Action',
                        'Scale': scale,
                        'Mean_Error': m['Mean_Error'],
                        'MAE': m['MAE'],
                        'N': m['N']
                    })

        if ccb_metrics is not None and len(ccb_metrics) > 0:
            for scale in ['original', 'rounded']:
                overall = ccb_metrics[
                    (ccb_metrics['scale'] == scale) &
                    (ccb_metrics['question'] == 'OVERALL')]
                if len(overall) > 0:
                    m = overall.iloc[0]
                    summary_rows.append({
                        'Task': 'CCB',
                        'Scale': scale,
                        'Mean_Error': m['Mean_Error'],
                        'MAE': m['MAE'],
                        'N': m['N']
                    })

        if mist_metrics is not None and len(mist_metrics) > 0:
            overall = mist_metrics[mist_metrics['Category'] == 'OVERALL']
            if len(overall) > 0:
                m = overall.iloc[0]
                summary_rows.append({
                    'Task': 'MIST',
                    'Scale': 'binary',
                    'Accuracy': m['Accuracy'],
                    'Precision': m['Precision'],
                    'Recall': m['Recall'],
                    'F1': m['F1'],
                    'N': m['N']
                })

        if mist_sdt is not None and len(mist_sdt) > 0:
            summary_rows.append({
                'Task': 'MIST_SDT',
                'Scale': 'sdt',
                'd_prime': mist_sdt['d_prime'],
                'c_bias': mist_sdt['c_bias'],
                'Hit_Rate': mist_sdt['Hit_Rate'],
                'False_Alarm_Rate': mist_sdt['False_Alarm_Rate'],
                'N': mist_sdt['N_True_Items'] + mist_sdt['N_False_Items']
            })

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    print(f"    Saved successfully!")


def save_group_analysis_to_excel(all_results, output_path):
    """Save group analysis results to Excel"""
    print(f"\n[Saving group analysis to: {output_path}]")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        if all_results['affect_science']:
            df = pd.concat(all_results['affect_science'], ignore_index=True)
            df_summary = df[df['scale'] == 'original'].copy()
            df_summary.to_excel(writer, sheet_name='AffectSci_ByGroup', index=False)
            df.to_excel(writer, sheet_name='AffectSci_ByGroup_AllScales', index=False)

        if all_results['affect_action']:
            df = pd.concat(all_results['affect_action'], ignore_index=True)
            df_summary = df[df['scale'] == 'original'].copy()
            df_summary.to_excel(writer, sheet_name='AffectAct_ByGroup', index=False)
            df.to_excel(writer, sheet_name='AffectAct_ByGroup_AllScales', index=False)

        if all_results['ccb']:
            df = pd.concat(all_results['ccb'], ignore_index=True)
            df_summary = df[df['scale'] == 'original'].copy()
            df_summary.to_excel(writer, sheet_name='CCB_ByGroup', index=False)
            df.to_excel(writer, sheet_name='CCB_ByGroup_AllScales', index=False)

        if all_results['mist']:
            df = pd.concat(all_results['mist'], ignore_index=True)
            df.to_excel(writer, sheet_name='MIST_ByGroup', index=False)

        # Summary sheet
        summary_rows = []

        for task_key, task_label in [('affect_science', 'Affect_Science'),
                                      ('affect_action', 'Affect_Action'),
                                      ('ccb', 'CCB')]:
            if all_results[task_key]:
                df = pd.concat(all_results[task_key], ignore_index=True)
                df = df[df['scale'] == 'original']
                for _, row in df.iterrows():
                    summary_rows.append({
                        'Task': task_label,
                        'Group_Var': row['group_var'],
                        'Group_Value': row['group_value'],
                        'Metric': 'MAE',
                        'Value': row['MAE'],
                        'N': row['N']
                    })

        if all_results['mist']:
            df = pd.concat(all_results['mist'], ignore_index=True)
            for _, row in df.iterrows():
                summary_rows.append({
                    'Task': 'MIST',
                    'Group_Var': row['group_var'],
                    'Group_Value': row['group_value'],
                    'Metric': 'F1',
                    'Value': row['F1'],
                    'N': row['N']
                })
                if 'd_prime' in row and pd.notna(row.get('d_prime')):
                    summary_rows.append({
                        'Task': 'MIST',
                        'Group_Var': row['group_var'],
                        'Group_Value': row['group_value'],
                        'Metric': 'd_prime',
                        'Value': row['d_prime'],
                        'N': row['N']
                    })

        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    print("    Saved successfully!")
