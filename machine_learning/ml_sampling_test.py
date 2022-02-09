# %%
import pandas as pd
import sqlalchemy
import matplotlib as mpl
import pickle
mpl.rcParams['figure.dpi'] = 300
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from flask import Blueprint, render_template
from website import db
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Create the engine to connect to the PostgreSQL database
engine = sqlalchemy.create_engine('sqlite:///C:\\Users\\Pontiff\\Desktop\\SIMPLON_DEV_IA_DOCUMENTS_THIBAUT_PERNET\\-G6-L-IA-au-service-de-la-s-curit-alimentaire\\website\\alim_confiance.db')

# Read raw_data from SQL table
raw_data_dataframe_query = pd.read_sql_table('raw_data', engine)
raw_data_dataframe = raw_data_dataframe_query

# Read inspection_data from SQL table
inspection_data_dataframe_query = pd.read_sql_table('inspection_data', engine)
inspection_data_dataframe = inspection_data_dataframe_query

feature = raw_data_dataframe[['store_industry','store_zipcode']]
label = inspection_data_dataframe['inspection_result']

print(type(feature))
print(feature.info())

feature_encoder = OneHotEncoder()
label_encoder = LabelEncoder()

feature = feature_encoder.fit_transform(feature)
label = label_encoder.fit_transform(label)
print(type(feature))
print(type(label))

""" # Random oversampling
# Create the sampler
oversampler = RandomOverSampler(sampling_strategy='auto')
feature_resampled, label_resampled = oversampler.fit_resample(feature, label)

# Compare label classes to check sampling result.
print(f'Label brut:\n{Counter(label)}')
print(f'Label resampled:\n{Counter(label_resampled)}')

# Create test and training variables from the oversampled dataframe
feature_train, feature_test, label_train, label_test = model_selection.train_test_split(feature_resampled, label_resampled, test_size=0.30, random_state=69) """

# SMOTE oversampling
""" seed = 69
smote = SMOTE(sampling_strategy='auto', random_state=seed, k_neighbors=7)
feature_resampled, label_resampled = smote.fit_resample(feature, label) """

# Random undersampling
""" # Create the sampler
oversampler = RandomUnderSampler(sampling_strategy='majority')
feature_resampled, label_resampled = oversampler.fit_resample(feature, label)

# Compare label classes to check sampling result.
print(f'Label brut:\n{Counter(label)}')
print(f'Label resampled:\n{Counter(label_resampled)}')

# Create test and training variables from the undersampled dataframe
feature_train, feature_test, label_train, label_test = model_selection.train_test_split(feature_resampled, label_resampled, test_size=0.30, random_state=69) """

""" # NearMiss undersampling
distribution_of_samples = {0:600, 1:123, 2:5000, 3:5000}
nearmiss = NearMiss(sampling_strategy = distribution_of_samples)
feature_resampled, label_resampled = nearmiss.fit_resample(feature, label) """

# Create test and training variables from the base dataframe
feature_train, feature_test, label_train, label_test = model_selection.train_test_split(feature, label, test_size=0.30, random_state=69)

# Create instance of RandomForestClassifier class with chosen training parameters
forest_classifier = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1)

# Build a random forest from the training set of data
forest_classifier.fit(feature_train, label_train)

# Predict a class for each feature
forest_label_pred = forest_classifier.predict(feature_test)

# Display classification report of Random Forest
print("Random Forest Classification Report")
print(metrics.classification_report(label_test, forest_label_pred, zero_division=0))

# Display confusion matrix
forest_matrix = metrics.plot_confusion_matrix(forest_classifier, feature_test, label_test)
forest_matrix.figure_.suptitle("Confusion Matrix (Random Forest)")
print(f"Confusion matrix:\n{forest_matrix.confusion_matrix}")
plt.show();