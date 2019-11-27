#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
#import seaborn as sns

pd.options.display.max_columns = None

import warnings
warnings.filterwarnings('ignore')

# Read the CSVs
dataset=pd.read_csv("E:/Trinity/Machine Learning/Kaggle/Project 2/tcd-ml-comp-201920-rec-alg-click-pred-group/Training.csv")
datatest=pd.read_csv("E:/Trinity/Machine Learning/Kaggle/Project 2/tcd-ml-comp-201920-rec-alg-click-pred-group/tcdml1920-rec-click-pred--test.csv")

# Replace String nulls to numpy nulls
dataset=dataset.replace('\\N',pd.np.nan)
datatest=datatest.replace('\\N',pd.np.nan)
datatest=datatest.replace('nA',pd.np.nan)

# Filter only Org_id 8 data
d8=dataset[dataset["organization_id"]==8]
d8test=datatest[datatest["organization_id"]==8] 
#len_train = len(d8)
#len_test  = len(d8test)

# Add a column to differentiate test and train data
d8["isTrain"] = 1
d8test["isTrain"] = 0

# Concat Test and Train datasets
d8 = pd.concat([d8, d8test], sort=False)

d8.isnull().sum()

# Drop unwanted columns
col_to_drop = ["recommendation_set_id", "user_id", "session_id", "document_language_provided", "year_published", 
               "number_of_authors", "first_author_id", "num_pubs_by_first_author", "app_version", "app_lang",
               "user_os", "user_os_version", "user_java_version", "user_timezone", "application_type", 
               "response_delivered", "rec_processing_time", "timezone_by_ip", "time_recs_recieved",
               "time_recs_displayed", "time_recs_viewed", "number_of_recs_in_set", "organization_id", 
               "item_type", "ctr", "clicks"]

d8.drop(col_to_drop, axis=1, inplace=True)

# Initialise label encoder 
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
scaler = preprocessing.MinMaxScaler()

# Preprocess each column
d8["query_identifier"] = d8["query_identifier"].str.split("|", expand=True,)[0]
d8["query_identifier"].fillna(d8["query_identifier"].mode()[0], inplace=True )
d8['query_identifier']= label_encoder.fit_transform(d8['query_identifier'])

d8["query_word_count"].fillna(d8["query_word_count"].mode()[0], inplace=True )
d8["query_word_count"] = d8["query_word_count"].astype(int)
d8['query_word_count'] = scaler.fit_transform(d8['query_word_count'].values.reshape(-1,1))

d8["query_char_count"].fillna(d8["query_char_count"].mode()[0], inplace=True )
d8['query_char_count'] = scaler.fit_transform(d8['query_char_count'].values.reshape(-1,1))

d8["query_detected_language"].fillna(d8["query_detected_language"].mode()[0], inplace=True )
d8['query_detected_language']= label_encoder.fit_transform(d8['query_detected_language'])

d8["query_document_id"].fillna(d8["query_document_id"].mode()[0], inplace=True )
d8['query_document_id'] = scaler.fit_transform(d8['query_document_id'].values.reshape(-1,1))

d8["abstract_word_count"].fillna(d8["abstract_word_count"].mode()[0], inplace=True )
d8['abstract_word_count'] = scaler.fit_transform(d8['abstract_word_count'].values.reshape(-1,1))

d8["abstract_char_count"].fillna(d8["abstract_char_count"].mode()[0], inplace=True )
d8['abstract_char_count'] = scaler.fit_transform(d8['abstract_char_count'].values.reshape(-1,1))

d8["abstract_detected_language"].fillna(d8["abstract_detected_language"].mode()[0], inplace=True )
d8['abstract_detected_language']= label_encoder.fit_transform(d8['abstract_detected_language'])

d8["request_received"].fillna(d8["request_received"].mode()[0], inplace=True )
d8['request_received']= label_encoder.fit_transform(d8['request_received'])
d8['request_received']= scaler.fit_transform(d8['request_received'].values.reshape(-1,1))

d8["country_by_ip"].fillna(d8["country_by_ip"].mode()[0], inplace=True )
d8['country_by_ip']= label_encoder.fit_transform(d8['country_by_ip'])

d8["local_time_of_request"].fillna(d8["local_time_of_request"].mode()[0], inplace=True )
d8['local_time_of_request']= label_encoder.fit_transform(d8['local_time_of_request'])
d8['local_time_of_request']= scaler.fit_transform(d8['local_time_of_request'].values.reshape(-1,1))

d8["local_hour_of_request"].fillna(d8["local_hour_of_request"].mode()[0], inplace=True )
d8['local_hour_of_request']= label_encoder.fit_transform(d8['local_hour_of_request'])

d8['algorithm_class']= label_encoder.fit_transform(d8['algorithm_class'])

d8["recommendation_algorithm_id_used"].fillna(d8["recommendation_algorithm_id_used"].mode()[0], inplace=True )

d8["cbf_parser"] = d8["cbf_parser"].replace(pd.np.nan, "no") 
d8['cbf_parser']= label_encoder.fit_transform(d8['cbf_parser'])

d8['search_title']= label_encoder.fit_transform(d8['search_title'])

d8['search_abstract']= label_encoder.fit_transform(d8['search_abstract'])

d8['search_keywords']= label_encoder.fit_transform(d8['search_keywords'])

d8['query_detected_language']= scaler.fit_transform(d8['query_detected_language'].values.reshape(-1,1))
d8['hour_request_received']= scaler.fit_transform(d8['hour_request_received'].values.reshape(-1,1))
d8['country_by_ip']= scaler.fit_transform(d8['country_by_ip'].values.reshape(-1,1))
d8['local_hour_of_request']= scaler.fit_transform(d8['local_hour_of_request'].values.reshape(-1,1))
d8['recommendation_algorithm_id_used']= scaler.fit_transform(d8['recommendation_algorithm_id_used'].values.reshape(-1,1))
d8['algorithm_class']= scaler.fit_transform(d8['algorithm_class'].values.reshape(-1,1))
d8['cbf_parser']= scaler.fit_transform(d8['cbf_parser'].values.reshape(-1,1))
d8['query_identifier']= scaler.fit_transform(d8['query_identifier'].values.reshape(-1,1))

# keep a copy, incase something happens after this!!!
d8copy = d8.copy()

# Split back Train data
x = d8[d8["isTrain"]==1]
y_full = x['set_clicked']
x_full = x.drop(columns=['set_clicked','isTrain'])

# Split train and holdout data (80-20)
from sklearn.model_selection import train_test_split#, cross_val_score, GridSearchCV
x_train, x_holdOut, y_train, y_holdOut = train_test_split(x_full, y_full, train_size=0.8, random_state=10)

# Split back Test data
x_t = d8[d8["isTrain"]==0]
x_test = x_t.drop(columns=['set_clicked','isTrain'])

# Train model on various parameter values - identify best performing parameter values

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

n_estimators = [500, 1000, 1500, 2000]
#depth = [3, 4, 5, 6, 7, 8, 9, 10]
for iters in n_estimators:
    model=RandomForestClassifier(n_estimators=iters, random_state=100)
    model.fit(x_train, y_train)  #,plot=True)
    ypred = model.predict(x_holdOut)
    print ("---------------------------Iterations    : " + str(iters))
    print ("---------------------------F1 Score      : " + str(f1_score(y_holdOut, ypred)))
    print ("-----------------------------------------------------------------------")


# CatBoostClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

iterations = [500, 1000]
learning_rate = [0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.5, 0.75, 1.0]
depth = [9, 10, 11, 12]

for iters in iterations:
    for lr in learning_rate:
        for dt in depth:
            model=CatBoostClassifier(iterations=iters, depth=dt, learning_rate=lr, verbose=False)
            model.fit(x_train, y_train)  #,plot=True)
            ypred = model.predict(x_holdOut)
            pred = pd.DataFrame(ypred)[0].astype(int)
            print ("---------------------------Iterations    : " + str(iters))
            print ("---------------------------Learning Rate : " + str(lr))
            print ("---------------------------Depth         : " + str(dt))
            print ("---------------------------F1 Score      : " + str(f1_score(y_holdOut, pred)))
            print ("-----------------------------------------------------------------------")


# ---------------------------Iterations    : 500
# ---------------------------Learning Rate : 0.2
# ---------------------------Depth         : 12
# --------------------------- F1 Score      : 0.9924812030075187
# -----------------------------------------------------------------------


# Train model with parameter values identified above
from sklearn.metrics import classification_report, confusion_matrix, f1_score
model=CatBoostClassifier(iterations=500, depth=5, learning_rate=0.2, verbose=False)
model.fit(x_train, y_train)  

# Final predict on test data
final_pred = model.predict(x_test)
pd.DataFrame(final_pred).to_csv("E:/Trinity/Machine Learning/Kaggle/Project 2/tcd-ml-comp-201920-rec-alg-click-pred-group/d1.csv")

