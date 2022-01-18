import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

alimconfiance_dataset_raw = pd.read_csv('./datasets/export_alimconfiance_raw.csv', sep=';')
alimconfiance_dataset_raw.fillna("_", inplace=True)
print(len(alimconfiance_dataset_raw))
industries = alimconfiance_dataset_raw['APP_Libelle_activite_etablissement'].astype(str)

industry_separator = "|"
loop_length = len(alimconfiance_dataset_raw)

for i in range(0,loop_length):
    row = alimconfiance_dataset_raw.iloc[i]
    industry = str(row['APP_Libelle_activite_etablissement'])
    industry_split = industry.split(industry_separator)
    if len(industry_split) > 1:
    #if industry.find(industry_separator):    
        for industry_single in industry_split:
            industry_single = industry_split[0]
            row['APP_Libelle_activite_etablissement'] = industry_single.format(row['APP_Libelle_activite_etablissement'])
            dataframe_test = pd.DataFrame(row)
            dataframe_test = pd.DataFrame.transpose(dataframe_test)
            alimconfiance_dataset_raw = alimconfiance_dataset_raw.append(dataframe_test)
            industry_split.pop(0)
            row['APP_Libelle_activite_etablissement'] = np.nan

alimconfiance_dataset_raw.dropna(inplace=True)
print(len(alimconfiance_dataset_raw)) 
#alimconfiance_dataset_raw.to_csv('./datasets/test.csv', index=False)      