import pandas as pd


df = pd.read_csv('/Users/vi507502/Documents/SBBench/AmbiguousBBQsimplifiedQid.csv')

zz = df.groupby('unique_question_id').size().reset_index(name='count')

zz = zz[zz['count']>2]

print(zz)
