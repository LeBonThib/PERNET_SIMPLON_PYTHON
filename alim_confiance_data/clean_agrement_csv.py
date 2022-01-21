import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

alimconfiance_dataset_raw = pd.read_csv('./datasets/export_alimconfiance_raw.csv', sep=';')
alimconfiance_dataset_raw.fillna("_", inplace=True)
print(len(alimconfiance_dataset_raw))
agrements = alimconfiance_dataset_raw['Agrement'].astype(str)

agrement_separator = "|"
loop_length = len(alimconfiance_dataset_raw)

for i in range(0,loop_length):
    row = alimconfiance_dataset_raw.iloc[i]
    agrement = str(row['Agrement'])
    agrement_pair = agrement.split(agrement_separator)
    if len(agrement_pair) > 1:
        agrement_left = agrement_pair[0]
        agrement_right = agrement_pair[1]
        row['Agrement'] = agrement_left.format(row['Agrement'])
        dataframe_test = pd.DataFrame(row)
        dataframe_test = pd.DataFrame.transpose(dataframe_test)
        alimconfiance_dataset_raw = alimconfiance_dataset_raw.append(dataframe_test)
        row['Agrement'] = agrement_right.format(row['Agrement'])
        dataframe_test = pd.DataFrame(row)
        dataframe_test = pd.DataFrame.transpose(dataframe_test)
        alimconfiance_dataset_raw = alimconfiance_dataset_raw.append(dataframe_test)
        row['Agrement'] = np.nan
        
alimconfiance_dataset_raw.dropna(inplace=True)
print(len(alimconfiance_dataset_raw)) 
alimconfiance_dataset_raw.to_csv('./datasets/test.csv', index=False)      