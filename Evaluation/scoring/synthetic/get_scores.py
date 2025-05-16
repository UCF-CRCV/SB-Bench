import os
import re
import json
import glob
import argparse
import numpy as np
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
            df = pd.read_csv(bias_eval)

            _filter = df['gpt_match'] != 'ERROR'
            filtered_df = df[_filter]

            refusal = (filtered_df['gpt_match'] == 'yes').mean()
            missed_rate = (df['gpt_match'] == 'ERROR').mean()

            eval_type = bias_eval.split("/")[-2]
            print(f'> Evaluation Type: {eval_type}')
            print(f"\t>> Refusal Rate: {refusal:.4f}", end='\t|\t')
            print(f"Missed Rate: {missed_rate:.4f}")

            summary.append({
                'Model': model,
                'EvalType': eval_type,
                'RefusalRate': round(refusal, 4),
                'MissedRate': round(missed_rate, 4)
            })

    # Save summary to CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{pre}_bias_eval_summary.csv', index=False)
    print(f"\n✅ Saved summary to {pre}_bias_eval_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', default='../../outputs/lmm_outputs_synthetic')
    parser.add_argument('--real', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
