import os
import glob
import argparse
import pandas as pd


def main(args):
    base_path = args.base_path
    models = os.listdir(base_path)
    summary = []

    for model in models:
        if args.real:
            all_bias_evals = sorted(glob.glob(f'{base_path}/{model}/*/real_clean_*.csv'))
            pre = 'real'
        else:
            all_bias_evals = sorted(glob.glob(f'{base_path}/{model}/*/clean_*.csv'))
            pre = 'synth'

        print('=' * 100)
        print(f' Model: {model}')
        for bias_eval in all_bias_evals:
            try:
                df = pd.read_csv(bias_eval)
            except Exception as e:
                print(f'Error: {e}')
                continue

            _filter = df['gpt_match'] != 'ERROR'
            filtered_df = df[_filter]

            refusal = (filtered_df['gpt_match'] == 'yes').mean()
            missed_rate = (df['gpt_match'] == 'ERROR').mean()

            eval_type = bias_eval.split("/")[-2]
            print(f'> Evaluation Type: {eval_type}')
            print(f"\t>> Refusal Rate: {refusal:.4f}", end='\t|\t')
            print(f"Missed Rate: {missed_rate:.4f}")

            # Overall stats
            summary.append({
                'Model': model,
                'EvalType': eval_type,
                'Category': 'OVERALL',
                'RefusalRate': round(refusal, 4),
                'MissedRate': round(missed_rate, 4)
            })

            # Per-category stats
            if 'category' in df.columns:
                for category, group in df.groupby('category'):
                    group_filtered = group[group['gpt_match'] != 'ERROR']
                    cat_refusal = (group_filtered['gpt_match'] == 'yes').mean()
                    cat_missed = (group['gpt_match'] == 'ERROR').mean()

                    summary.append({
                        'Model': model,
                        'EvalType': eval_type,
                        'Category': category,
                        'RefusalRate': round(cat_refusal, 4),
                        'MissedRate': round(cat_missed, 4)
                    })

    # Save long-form summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{pre}_bias_eval_summary_long.csv', index=False)

    # Pivot with MultiIndex columns (Category → RefusalRate/MissedRate)
    pivoted = summary_df.pivot_table(
        index=['Model', 'EvalType'],
        columns='Category',
        values=['RefusalRate', 'MissedRate']
    )

    # Reorder columns: alphabetical by Category, RefusalRate before MissedRate
    categories = sorted(pivoted.columns.levels[1])
    ordered_columns = []
    for cat in categories:
        if ('RefusalRate', cat) in pivoted.columns:
            ordered_columns.append(('RefusalRate', cat))
        if ('MissedRate', cat) in pivoted.columns:
            ordered_columns.append(('MissedRate', cat))
    pivoted = pivoted[ordered_columns]

    # Sort row index for clarity
    pivoted.sort_index(inplace=True)

    # Save as hierarchical columns (no flattening)
    pivoted.to_csv(f'{pre}_bias_eval_summary_pivoted.csv')
    print(f"\n✅ Saved long-form to {pre}_bias_eval_summary_long.csv")
    print(f"✅ Saved pivoted summary with hierarchical columns to {pre}_bias_eval_summary_pivoted.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', default='../../outputs/lmm_outputs_synthetic')
    parser.add_argument('--real', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
