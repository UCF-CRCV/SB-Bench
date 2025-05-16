import json
import os
import re
import glob
import pandas as pd
from tqdm import tqdm

# 1) (Optional) load your original CSV if you need it later
df_original = pd.read_csv('hf_valid_v2.csv')

# 2) find all JSON outputs
json_files = glob.glob('Outputs/*.json')

# Create an empty list to store DataFrames
dfs = []

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

# Iterate through each JSON file
for file in json_files:
    data = read_json_safe(file)
    if data is not None:
        df = pd.DataFrame(data)  # Convert JSON data into DataFrame
        dfs.append(df)
    else:
        print(f"Skipping malformed file: {file}")

# Combine all DataFrames into one if any files were successfully read
if dfs:
    gpt_output = pd.concat(dfs, ignore_index=True)
else:
    print("No JSON files found or could be read in the specified directory.")
    exit()

# Function to split the output into reason, confidence, and approval
def split_output(generated_text):
    try:
        match = re.findall(r'```(.*?)```', generated_text, re.DOTALL)
        if match:
            generated_text = match[0][4:].strip()
        parsed = json.loads(generated_text)
        LMM_reason = parsed.get("reason", "").strip()
        LMM_confidence = str(parsed.get("confidence", "")).strip()
        LMM_approved = str(parsed.get("answer", "")).strip().lower()
    except Exception as e:
        LMM_reason = f"Parsing failed: {str(e)} | Raw output: {generated_text}"
        LMM_confidence = -1
        LMM_approved = "ERROR"
    return LMM_reason, LMM_confidence, LMM_approved

# Apply split_output and create new columns
gpt_output[['GPT_reason', 'GPT_confidence', 'GPT_rejected']] = gpt_output['output'].apply(split_output).apply(pd.Series)

# Save the filtered gpt_output
gpt_output.to_csv('filtered_gpt_images.csv', index=False)

# Filter out rows where GPT_rejected is 'ERROR' (or handle as needed)
gpt_output = gpt_output[gpt_output['GPT_rejected'] != "ERROR"]
gpt_output = gpt_output[gpt_output['GPT_rejected'] == 'false']
gpt_output = gpt_output.reset_index(drop=True)

# Merge with the original dataframe on 'id' and 'image_id'
df_original['id'] = df_original['id'].astype(str)
gpt_output['image_id'] = gpt_output['image_id'].astype(str)

# Perform the merge
df = pd.merge(df_original, gpt_output, left_on='id', right_on='image_id')

# Drop unnecessary columns
df_nonneg = df.drop(columns=['image_id', 'output'])

# Save the final result
df_nonneg.to_csv('nonneg.csv', index=False)


# Merge negative questions again!
df_neg    = pd.read_csv('hf_valid_v2.csv')

# --- 2. Merge nonneg into neg on unique_question_id ---
#    suffixes: keep original neg columns untouched, nonneg columns get "_nonneg"
df = df_neg.merge(
    df_nonneg,
    on='file_name',
    how='inner',
    suffixes=('', '_nonneg')
)

# --- 3. For each of your shared columns, fill neg NaNs from nonneg ---
shared_cols = ['question', 'context']  # adjust if your column names differ
for col in shared_cols:
    df[col] = df[col].fillna(df[f'{col}_nonneg'])

# --- 4. Drop the extra "_nonneg" columns now that we’ve filled them in ---
df = df.drop(columns=[f'{col}_nonneg' for col in shared_cols])

# --- 5. (Optional) sanity‐check that every ID got filled where neg was missing ---
missing = df[df[shared_cols].isna().any(axis=1)]
if not missing.empty:
    print("Still missing for some IDs:", missing['id'].tolist())

# Now `df` has your neg questions/contexts, with any gaps filled in from the nonneg DataFrame.
df = df.dropna(subset=['file_name'])
df = df[['file_name', 'id', 'category', 'additional_metadata', 'question_polarity', 'context', 'question', 'ans0', 'ans1' , 'ans2', 'label']]
df.to_csv('hf_valid_v3.csv', index=False)
