import json
import glob
import pandas as pd

df = pd.read_csv('data/AmbiguousBBQ.csv')
outputs = glob.glob('Outputs/*.json')

if len(outputs)>1:
    print('This code doesn\'t support multiple files yet...')
    exit()

gpt_output = pd.read_json(outputs[0],)

df = pd.merge(df, gpt_output, left_on='uid', right_on='image_id')
df[['simplified_reason', 'simplified_context']] = df['output'].apply(
    lambda x: pd.Series({
        'simplified_reason': json.loads(x)['reason'],
        'simplified_context': json.loads(x)['simplified_context']
    })
)

df = df.drop(columns=['image_id', 'output'])

df.to_csv('AmbiguousBBQsimplified.csv', index=False)
