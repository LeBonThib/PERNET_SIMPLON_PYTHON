#%%
#   IMPORT DES PACKAGES ET LIBS
#
from email.mime import base
from locale import normalize
from xml.etree.ElementInclude import include
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#%%
#   OBJECTIF
#
print("- Comprendre la nature des données")
print("- Développer une premiere stratégie de modélisation")

#%%
#   IMPORT DU DATASET
#
base_dataset = 'C:/Users/Pontiff/Desktop/SIMPLON_DEV_IA_DOCUMENTS_THIBAUT_PERNET/python/datasets/export_alimconfiance_raw.csv'
base_dataframe = pd.read_csv(base_dataset, sep=";")
# base_dataframe = pd.read_tsv(base_dataset)
# base_dataframe = pd.read_excel(base_dataset)
# base_dataframe = pd.read_json(base_dataset)
# base_dataframe = pd.read_sql(base_dataset)

#%%
#   COUP D'OEIL RAPIDE SUR LE DATASET
#
print(f'Informations sur le dataset:')
print(base_dataframe.info())
print(f'Cinq premières lignes du dataset:')
base_dataframe.head()

#%%
#   ANALYSE DE FORME DES DONNEES - PARAMÉTRAGE DE LA TARGET
#
target_label = 'Synthese_eval_sanit'
base_target = base_dataframe[target_label]


#%%
#   CHECKLIST DE BASE
#
print('Analyse de Forme:')
print(f'dimensions:\n{base_dataframe.shape}')
print('types de variables:')
print(f"qualitatives:\n {len(base_dataframe.select_dtypes(include=['object']).columns)}")
print(f"quantitatives:\n {len(base_dataframe.select_dtypes(include=['float','int','uint']).columns)}")

#%%
#   ANALYSE DE FORME DES DONNEES - DTYPES & HEATMAP DES NaN
#
base_dataframe.dtypes.value_counts().plot(kind='pie', title='Répartition des types dans le dataset', label='')
plt.figure().suptitle('Heatmap des cellules vides du dataset')
sns.heatmap(base_dataframe.isna(), cbar=False)

# %%
#   ANALYSE DE FORME DES DONNEES - POURCENTAGE DE NaN PAR COLONNE
#
(base_dataframe.isna().sum()/base_dataframe.shape[0]).sort_values(ascending=False)

# %%
#   ANALYSE DE FOND DES DONNEES - ELIMINATION DES COLONNES INUTILES (>90% NaN)
#
base_dataframe = base_dataframe[base_dataframe.columns[base_dataframe.isna().sum()/base_dataframe.shape[0] <0.9]]

# %%
#   ANALYSE DE FOND DES DONNEES - HEATMAP DES NaN (POST DROP)
#
plt.figure(figsize=(20,10))
sns.heatmap(base_dataframe.isna(), cbar=False)

# %%
#   ANALYSE DE FOND DES DONNEES - EXAMEN DE LA TARGET
#
print(f'Répartition des valeurs dans la cible\n{base_target.value_counts(normalize=True)}')

# %%
#   ANALYSE DE FOND DES DONNEES - DIAGRAMME DE DISTRIBUTION (SI VALEUR NUMÉRIQUE)
#
""" for col in base_dataframe.select_dtypes('float'):
    plt.figure()
    sns.displot(base_dataframe[col]) """

# %%
#   ANALYSE DE FOND DES DONNEES - HISTOGRAMES DES VARIABLES CONTINUES (SI VALEUR NUMÉRIQUE)
#
""" for col in base_dataframe.select_dtypes('float'):
    plt.figure()
    sns.displot(base_dataframe[col]) """

#%%
#   PRÉPARATION À L'ENCODAGE DES FEATURES ET DU LABEL POUR VISUALISATION MATHÉMATIQUE
#
# Assign data from the dataframe to the features and label variables by locating the columns.
features = base_dataframe.loc[:, base_dataframe.columns != target_label]
label = base_target

# Cast features as string to unify type prior to encoding.
features = features.astype(str)

#%%
#   ON ENCODE LE LABEL VIA LABELENCODER ET LES FEATURES VIA ONEHOTENCODER
#
# Import encoders.
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Create the encoders.
feature_encoder = OneHotEncoder()
label_encoder = LabelEncoder()

# Apply the encoder.
features = feature_encoder.fit_transform(features)
label = label_encoder.fit_transform(label)

#%%
#   ON SPLIT LES VARIABLES LABEL ET FEATURES EN LES SOUS-COMPOSANTES DE TEST ET DE TRAINING (APRÈS RESAMPLING)
#
# Create test and training variables from the oversampled dataframe
feature_train, feature_test, label_train, label_test = model_selection.train_test_split(features, label, test_size=0.30, random_state=69) 

#%%
#   ON PRÉPARE LE CLASSIFIER POUR GÉNERER UN ARBRE DE DECISION
#
# Create instance of DecisionTreeClassifier class with chosen training parameters
tree_classifier = DecisionTreeClassifier(max_leaf_nodes=4, max_depth=10)

# Build a decision tree from the training set of data
tree_classifier.fit(feature_train, label_train)

# Predict a class for each feature
tree_label_pred = tree_classifier.predict(feature_test)

#%%
#   ON AFFICHE L'ARBRE DE DÉCISION
#
# Display decision tree 
tree.plot_tree(tree_classifier);

#%%
#   ON EXTRAIT LES SCORES DU MODELE (DECISION TREE)
#
# Add return of score function to variable for further exploitation of data
test_tree_score = tree_classifier.score(feature_test, label_test)
train_tree_score = tree_classifier.score(feature_train, label_train)

# Compare training and test scores
print(f'Score feature_test, label_test:\n{test_tree_score}')
print(f'Score feature_train, label_train:\n{train_tree_score}')

# %%
#   ON AFFICHE LA MATRICE DE CONFUSION (DECISION TREE)
#
# Display classification report of Decision Tree
print("Decision Tree Classification Report")
print(metrics.classification_report(label_test, tree_label_pred, zero_division=0))

# Display confusion matrix
tree_matrix = metrics.plot_confusion_matrix(tree_classifier, feature_test, label_test)
tree_matrix.figure_.suptitle("Confusion Matrix (Decision Tree)")
print(f"Confusion matrix:\n{tree_matrix.confusion_matrix}")
plt.show();

# %%
#   ON PRÉPARE LE CLASSIFIER POUR GÉNERER UNE RANDOM FOREST
#
# Create instance of RandomForestClassifier class with chosen training parameters
forest_classifier = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=-1)

# Build a random forest from the training set of data
forest_classifier.fit(feature_train, label_train)

# Predict a class for each feature
forest_label_pred = forest_classifier.predict(feature_test)

# %%
#   ON EXTRAIT LES SCORES DU MODELE (RANDOM FOREST)
#
# Display classification report of Random Forest
print("Random Forest Classification Report")
print(metrics.classification_report(label_test, forest_label_pred, zero_division=0))

# %%
#   ON AFFICHE LA MATRICE DE CONFUSION (RANDOM FOREST)
#
# Display confusion matrix
forest_matrix = metrics.plot_confusion_matrix(forest_classifier, feature_test, label_test)
forest_matrix.figure_.suptitle("Confusion Matrix (Random Forest)")
print(f"Confusion matrix:\n{forest_matrix.confusion_matrix}")
plt.show();
