import json
import os
import re
import glob
import pandas as pd
from tqdm import tqdm
import argparse


# Function to read the JSON file with error handling for malformed content
def read_json_safe(file_path):
    try:
        with open(file_path, 'r') as f:
            # Try reading the JSON file
            data = json.load(f)  # Load the entire JSON content
        return data
    except json.JSONDecodeError as e:
        print(f"Error reading file {file_path}: {e}")
        return None  # Return None if the file is malformed
    except Exception as e:
        print(f"Unexpected error reading file {file_path}: {e}")
        return None

def split_output(generated_text):
    try:
        match = re.findall(r'```(.*?)```', generated_text, re.DOTALL)
        if match:
            if match[0][:4]=='json':
                generated_text = match[0][4:].strip()    
            else:
                generated_text = match[0].strip()
        parsed = json.loads(generated_text)
        LMM_answer = str(parsed.get("answer", "")).strip()
        LMM_predicted_answer = parsed.get("predicted_answer", "").strip()
    except Exception as e:
        LMM_answer = "ERROR"
        LMM_predicted_answer = f"Parsing failed: {str(e)} | Raw output: {generated_text}"
    return LMM_answer, LMM_predicted_answer


def main(base_path):
    # Read the raw JSON file into a DataFrame
    file_df = pd.read_json('file_ids.json')
    batch_df = pd.read_json('batch_ids.json')

    file_df[0] = file_df[0].apply(lambda x: json.loads(x))
    file_df = pd.json_normalize(file_df[0])

    merged = pd.merge(file_df, batch_df, left_on='id', right_on='file_id')

    for i, row in tqdm(merged[['batch_id', 'filename']].iterrows()):
        batch_id = row.get('batch_id')
        filename = row.get('filename')

        eval_type = filename.split('_')[0]  #mcq/oe
        model = filename.split('.jsonl')[0][len(eval_type)+1:]

        data = read_json_safe(os.path.join('Outputs', batch_id+'.json'))
        gpt_df = pd.DataFrame(data)
        origin_df = pd.read_csv(f'{base_path}/{model}/{eval_type}/results.csv')

        final = pd.merge(origin_df, gpt_df, left_on='id', right_on='custom_id', how='left')
        final[['gpt_match', 'predicted_answer']] = final['output'].apply(split_output).apply(pd.Series)
        final.loc[final['gpt_match'] == '', 'gpt_match'] = 'ERROR'

        output_path = f'{base_path}/{model}/{eval_type}/'
        os.makedirs(output_path, exist_ok=True)
        final.to_csv(os.path.join(output_path, 'clean_results.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', default='../../../outputs/lmm_outputs_base_llm')

    args = parser.parse_args()


    base_path = args.base_path

    main(base_path)
