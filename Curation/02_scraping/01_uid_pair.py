import pandas as pd

# Load your CSV
df = pd.read_csv("../simplify/AmbiguousBBQsimplified.csv")

# Check if there is a 'context' column to group by
if 'context' not in df.columns:
    raise ValueError("Expected a 'context' column to group by.")

# Create a unique ID per context group
df['unique_question_id'] = df.groupby('context').ngroup()

# Optional: to make it more readable (e.g., Q1, Q2, ...)
df['unique_question_id'] = df['unique_question_id'].apply(lambda x: f"InterQ{x+1}")


# ----------------------------------
# Group by 'unique_question_id' to get counts and filter where count is greater than 2 (i.e., 4 counts)
zz = df.groupby('unique_question_id').size().reset_index(name='count')
zz = zz[zz['count'] == 4]  # Filter to rows with 4 occurrences of the same unique_question_id

# Create a temporary dataframe where we will modify the rows with 4 occurrences
temp_df = df[df['unique_question_id'].isin(zz['unique_question_id'])].copy()

# Subtract 1 from 'uid' where 'question_polarity' is non-negative
temp_df['subtracted_uid'] = temp_df.apply(
    lambda row: row['uid'] - 1 if row['question_polarity'] == 'nonneg' else row['uid'],
    axis=1
)

temp_df['unique_question_id'] = temp_df.groupby('subtracted_uid').ngroup()

# Create a new unique_question_id in the format UpdtdQ{XXXX} using subtracted_uid
temp_df['new_unique_question_id'] = temp_df.apply(
    lambda row: f"UpdtdInterQ{row['unique_question_id']}", axis=1
)

# Merge this temporary dataframe with the original dataframe based on 'uid'
df = df.merge(temp_df[['uid', 'new_unique_question_id']], on='uid', how='left')

# Update the original 'unique_question_id' only where we have a new 'new_unique_question_id'
df['unique_question_id'] = df['new_unique_question_id'].fillna(df['unique_question_id'])

df = df.drop(columns=['new_unique_question_id'])

# Save the updated dataframe to a new CSV
df.to_csv('AmbiguousBBQsimplifiedQid.csv', index=False)

print("Unique question IDs updated.")