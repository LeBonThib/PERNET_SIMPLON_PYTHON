import pandas as pd

alimconfiance_dataset = pd.read_csv('./datasets/export_alimconfiance_raw.csv', sep=';')
alimconfiance_dataset.fillna("_", inplace=True)
alimconfiance_dataset_clean_raw_import = alimconfiance_dataset.copy()
alimconfiance_dataset_clean_inspection_import = alimconfiance_dataset.copy()

alimconfiance_dataset_clean_raw_import.rename(columns={"APP_Libelle_etablissement":"nom_etablissement","SIRET":"numero_siret","Adresse_2_UA":"adresse_etablissement","Code_postal":"code_postal_etablissement",
"Libelle_commune":"ville_etablissement", "APP_Libelle_activite_etablissement":"activite_etablissement","Agrement":"agrement_etablissement","geores":"geoloc_etablissement",
"filtre":"filtre_etablissement","ods_type_activite":"ods_etablissement"}, inplace=True)
alimconfiance_dataset_clean_raw_import.drop(columns=['Numero_inspection', 'Date_inspection','Synthese_eval_sanit'], inplace=True)
alimconfiance_dataset_clean_raw_import.to_csv('./datasets/export_alimconfiance_clean_raw_import.csv')

alimconfiance_dataset_clean_raw_import = pd.read_csv('./datasets/export_alimconfiance_clean_raw_import.csv', sep=',')
alimconfiance_dataset_clean_raw_import.columns.values[0] = 'index'
alimconfiance_dataset_clean_raw_import["index"] = alimconfiance_dataset_clean_raw_import["index"] + 1
alimconfiance_dataset_clean_raw_import.to_csv('./datasets/export_alimconfiance_clean_raw_import.csv', index=False)

alimconfiance_dataset_clean_inspection_import.rename(columns={"Numero_inspection":"numero_inspection","Date_inspection":"date_inspection","Synthese_eval_sanit":"resultat_inspection"}, inplace=True)
alimconfiance_dataset_clean_inspection_import.drop(columns=['APP_Libelle_etablissement','SIRET','Adresse_2_UA','Code_postal',
'Libelle_commune','APP_Libelle_activite_etablissement','Agrement','geores','filtre','ods_type_activite'], inplace=True)
alimconfiance_dataset_clean_inspection_import.to_csv('./datasets/export_alimconfiance_clean_inspection_import.csv', index=False)