import json
import pandas as pd
import glob

df = pd.read_csv('../03_filtering/filtered_images_with_BBQ_GPT_verified.csv')

outputs = glob.glob('Outputs/*.json')

if len(outputs)>1:
    print('This code doesn\'t support multiple files yet...')
    exit()

gpt_output = pd.read_json(outputs[0])

df = pd.merge(df, gpt_output, left_on='unique_question_id', right_on='image_id')
df[['anonymized_reason', 'anonymized_context']] = df['output'].apply(
    lambda x: pd.Series({
        'anonymized_reason': json.loads(x)['reason'],
        'anonymized_context': json.loads(x)['anonymized_text']
    })
)

df = df.drop(columns=['image_id', 'output'])

df.to_csv('filtered_anonymized_GPT_verified.csv', index=False)
