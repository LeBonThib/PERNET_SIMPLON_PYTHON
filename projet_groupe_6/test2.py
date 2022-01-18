import pandas as pd

alimconfiance_dataset_clean_raw_import = pd.read_csv('./datasets/export_alimconfiance_clean_raw_import.csv', sep=',')
agrements = alimconfiance_dataset_clean_raw_import['agrement_etablissement'].astype(str)

rot = alimconfiance_dataset_clean_raw_import.iloc[[0]]
newdf = pd.DataFrame(rot)
print(newdf)