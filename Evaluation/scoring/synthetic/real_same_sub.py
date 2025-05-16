import os
import glob
import pandas as pd
import re

def main(args):
    base_path = args.base_path
    real_path = args.real_path
    
    models = os.listdir(base_path)

    for model in models:
        all_bias_evals = sorted(glob.glob(f'{base_path}/{model}/*/clean_*.csv'))

        for bias_eval in all_bias_evals:
            eval_type = bias_eval.split("/")[-2]
            print(f"\nProcessing: Model = {model} | EvalType = {eval_type}")

            df = pd.read_csv(bias_eval)

            # Construct path to origin results
            origin_path = f'{real_path}/{model}/{eval_type}/clean_results.csv'
            if not os.path.exists(origin_path):
                print(f"⚠️ Missing file: {origin_path}")
                continue

            origin_df = pd.read_csv(origin_path)

            # Step 1: Build `cross_id`
            cats = sorted(df['category'].unique())
            category_mapping = {cat: f"{i+1:02d}" for i, cat in enumerate(cats)}
            df['cat_code'] = df['category'].map(category_mapping)

            df['qidx_str'] = df['question_index'].astype(str).str.zfill(2)
            df['exid_str'] = df['example_id'].astype(str).str.zfill(4)

            polarity_mapping = {
                'nonneg': '1',
                'neg': '2',
            }
            df['pol_code'] = df['question_polarity'].map(polarity_mapping)

            df['cross_id'] = (
                df['cat_code'] + '_' +
                df['qidx_str'] + '_' +
                df['exid_str'] + '_' +
                df['pol_code']
            )

            # Step 2: Match rows from origin_df where id starts with any cross_id
            origin_df['id'] = origin_df['id'].astype(str)
            cross_ids = df['cross_id'].unique()

            # Build regex pattern: ^(cross1|cross2|cross3)_
            pattern = '^(' + '|'.join(re.escape(cid) for cid in cross_ids) + ')_'
            matched_rows = origin_df[origin_df['id'].str.match(pattern)]

            print(f"✅ Matched {len(matched_rows)} rows out of {len(origin_df)} total in origin_df")

            # Optional: save matched rows if needed
            matched_rows.to_csv(f'{base_path}/real_{model}/{eval_type}/clean_results.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', default='../../outputs/lmm_outputs_synthetic')
    parser.add_argument('--real_path', default='../../outputs/lmm_outputs')
    

    args = parser.parse_args()
    main(args)
