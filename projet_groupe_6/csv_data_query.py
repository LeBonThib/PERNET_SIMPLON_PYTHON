import pandas as pd
pd.set_option("display.max_rows", None)
alimconfiance_dataset = pd.read_csv('./datasets/export_alimconfiance.csv', sep=';')
alimconfiance_dataset_test = alimconfiance_dataset.copy()

#print(alimconfiance_dataset.head())

alimconfiance_columns = list(alimconfiance_dataset.columns)
print(alimconfiance_columns)

alimconfiance_columns_count = alimconfiance_dataset['Agrement']
print(alimconfiance_columns_count.value_counts(ascending=False))
print(alimconfiance_columns_count.count())

# modification du csv pour supprimer les doublons
# modification du csv pour rassembler les catégories mixes dans une des "grandes" catégories (e.g: rayon ... > grande distribution)