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
    """保存原始数据（每人每题每次）到Excel"""
    print(f"\n[Saving raw data to Excel: {output_path}]")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for task in df_raw['task'].unique():
            df_task = df_raw[df_raw['task'] == task].copy()
            df_task = df_task.sort_values(['row_id', 'repeat', 'question'])

            sheet_name = f'Raw_{task.capitalize()}'
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


def save_to_excel(belief_detail, belief_metrics,
                  policy_detail, policy_metrics,
                  share_detail, share_metrics, share_cm,
                  trees_detail, trees_results,
                  output_path,
                  belief_correlation=None,
                  policy_correlation=None):
    """保存聚合评估结果到Excel"""
    print(f"\n[Saving aggregated results to Excel: {output_path}]")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        if belief_detail is not None and len(belief_detail) > 0:
            belief_detail.to_excel(writer, sheet_name='Belief_Detail', index=False)

        if belief_metrics is not None and len(belief_metrics) > 0:
            belief_metrics.to_excel(writer, sheet_name='Belief_Metrics', index=False)

        if policy_detail is not None and len(policy_detail) > 0:
            policy_detail.to_excel(writer, sheet_name='Policy_Detail', index=False)

        if policy_metrics is not None and len(policy_metrics) > 0:
            policy_metrics.to_excel(writer, sheet_name='Policy_Metrics', index=False)

        if share_detail is not None and len(share_detail) > 0:
            share_detail.to_excel(writer, sheet_name='Share_Detail', index=False)

        if share_metrics is not None:
            pd.DataFrame([share_metrics]).to_excel(writer, sheet_name='Share_Metrics', index=False)

        if share_cm is not None:
            labels = ['No', 'Yes'][:share_cm.shape[0]]
            pd.DataFrame(share_cm,
                         index=[f'True:{l}' for l in labels],
                         columns=[f'Pred:{l}' for l in labels]).to_excel(writer, sheet_name='Share_ConfusionMatrix')

        if trees_detail is not None and len(trees_detail) > 0:
            trees_detail.to_excel(writer, sheet_name='Trees_Detail', index=False)

        if trees_results is not None:
            if 'continuous' in trees_results:
                pd.DataFrame([trees_results['continuous']]).to_excel(
                    writer, sheet_name='Trees_Continuous', index=False)

            if 'binary' in trees_results:
                pd.DataFrame([trees_results['binary']]).to_excel(
                    writer, sheet_name='Trees_Binary_Metrics', index=False)

            if 'binary_cm' in trees_results:
                labels = ['NotPlanted', 'Planted'][:trees_results['binary_cm'].shape[0]]
                pd.DataFrame(trees_results['binary_cm'],
                             index=[f'True:{l}' for l in labels],
                             columns=[f'Pred:{l}' for l in labels]).to_excel(
                    writer, sheet_name='Trees_Binary_CM')

            if 'ternary' in trees_results:
                pd.DataFrame([trees_results['ternary']]).to_excel(
                    writer, sheet_name='Trees_Ternary_Metrics', index=False)

            if 'ternary_cm' in trees_results:
                labels = ['None', 'Some', 'Full'][:trees_results['ternary_cm'].shape[0]]
                pd.DataFrame(trees_results['ternary_cm'],
                             index=[f'True:{l}' for l in labels],
                             columns=[f'Pred:{l}' for l in labels]).to_excel(
                    writer, sheet_name='Trees_Ternary_CM')

            if 'distribution' in trees_results:
                dist = trees_results['distribution']
                dist_rows = [{'trees': i, 'true': dist['true'].get(i, 0), 'pred': dist['pred'].get(i, 0)}
                             for i in range(9)]
                pd.DataFrame(dist_rows).to_excel(writer, sheet_name='Trees_Distribution', index=False)

        # Correlation sheets
        if belief_correlation is not None and len(belief_correlation) > 0:
            belief_correlation.to_excel(writer, sheet_name='Belief_Correlation', index=False)
        if policy_correlation is not None and len(policy_correlation) > 0:
            policy_correlation.to_excel(writer, sheet_name='Policy_Correlation', index=False)

        # Summary
        summary_rows = []

        if belief_metrics is not None and len(belief_metrics) > 0:
            for scale in ['original', 'decimal', 'discrete']:
                overall = belief_metrics[(belief_metrics['scale'] == scale) &
                                         (belief_metrics['question'] == 'OVERALL')]
                if len(overall) > 0:
                    m = overall.iloc[0]
                    summary_rows.append({
                        'Task': 'Belief',
                        'Scale': scale,
                        'Mean_Error': m['Mean_Error'],
                        'MAE': m['MAE'],
                        'N': m['N']
                    })

        if policy_metrics is not None and len(policy_metrics) > 0:
            for scale in ['original', 'decimal', 'discrete']:
                overall = policy_metrics[(policy_metrics['scale'] == scale) &
                                         (policy_metrics['question'] == 'OVERALL')]
                if len(overall) > 0:
                    m = overall.iloc[0]
                    summary_rows.append({
                        'Task': 'Policy',
                        'Scale': scale,
                        'Mean_Error': m['Mean_Error'],
                        'MAE': m['MAE'],
                        'N': m['N']
                    })

        if share_metrics is not None:
            summary_rows.append({
                'Task': 'Share',
                'Scale': 'binary',
                'Accuracy': share_metrics['Accuracy'],
                'Precision': share_metrics['Precision'],
                'Recall': share_metrics['Recall'],
                'F1': share_metrics['F1'],
                'N': share_metrics['N']
            })

        if trees_results is not None:
            if 'continuous' in trees_results:
                m = trees_results['continuous']
                summary_rows.append({
                    'Task': 'Trees',
                    'Scale': 'continuous',
                    'Mean_Error': m['Mean_Error'],
                    'MAE': m['MAE'],
                    'N': m['N']
                })

            if 'binary' in trees_results:
                m = trees_results['binary']
                summary_rows.append({
                    'Task': 'Trees',
                    'Scale': 'binary',
                    'Accuracy': m['Accuracy'],
                    'Precision': m['Precision'],
                    'Recall': m['Recall'],
                    'F1': m['F1'],
                    'N': m['N']
                })

            if 'ternary' in trees_results:
                m = trees_results['ternary']
                summary_rows.append({
                    'Task': 'Trees',
                    'Scale': 'ternary',
                    'Accuracy': m['Accuracy'],
                    'Precision': m['Precision'],
                    'Recall': m['Recall'],
                    'F1': m['F1'],
                    'N': m['N']
                })

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    print(f"    Saved successfully!")


def save_group_analysis_to_excel(all_results, output_path):
    """保存分组分析结果到Excel"""
    print(f"\n[Saving group analysis to: {output_path}]")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        if all_results['belief']:
            df_belief = pd.concat(all_results['belief'], ignore_index=True)
            df_belief_summary = df_belief[df_belief['scale'] == 'original'].copy()
            df_belief_summary.to_excel(writer, sheet_name='Belief_ByGroup', index=False)
            df_belief.to_excel(writer, sheet_name='Belief_ByGroup_AllScales', index=False)

        if all_results['policy']:
            df_policy = pd.concat(all_results['policy'], ignore_index=True)
            df_policy_summary = df_policy[df_policy['scale'] == 'original'].copy()
            df_policy_summary.to_excel(writer, sheet_name='Policy_ByGroup', index=False)
            df_policy.to_excel(writer, sheet_name='Policy_ByGroup_AllScales', index=False)

        if all_results['share']:
            df_share = pd.concat(all_results['share'], ignore_index=True)
            df_share.to_excel(writer, sheet_name='Share_ByGroup', index=False)

        if all_results['trees']:
            df_trees = pd.concat(all_results['trees'], ignore_index=True)
            df_trees.to_excel(writer, sheet_name='Trees_ByGroup', index=False)

        summary_rows = []

        for task in ['belief', 'policy']:
            if all_results[task]:
                df = pd.concat(all_results[task], ignore_index=True)
                df = df[df['scale'] == 'original']
                for _, row in df.iterrows():
                    summary_rows.append({
                        'Task': task.capitalize(),
                        'Group_Var': row['group_var'],
                        'Group_Value': row['group_value'],
                        'Metric': 'MAE',
                        'Value': row['MAE'],
                        'N': row['N']
                    })

        if all_results['share']:
            df = pd.concat(all_results['share'], ignore_index=True)
            for _, row in df.iterrows():
                summary_rows.append({
                    'Task': 'Share',
                    'Group_Var': row['group_var'],
                    'Group_Value': row['group_value'],
                    'Metric': 'F1',
                    'Value': row['F1'],
                    'N': row['N']
                })

        if all_results['trees']:
            df = pd.concat(all_results['trees'], ignore_index=True)
            for _, row in df.iterrows():
                summary_rows.append({
                    'Task': 'Trees',
                    'Group_Var': row['group_var'],
                    'Group_Value': row['group_value'],
                    'Metric': 'MAE',
                    'Value': row['MAE'],
                    'N': row['N']
                })

        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    print("    Saved successfully!")
