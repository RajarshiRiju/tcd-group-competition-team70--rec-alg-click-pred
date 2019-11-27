#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


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

# Filter only Org_id 1 data
d1=dataset[dataset["organization_id"]==1]
d1test=datatest[datatest["organization_id"]==1] 
len_train = len(d1)
len_test  = len(d1test)

# Add a column to differentiate test and train data
d1["isTrain"] = 1
d1test["isTrain"] = 0

# Concat Test and Train datasets
d1 = pd.concat([d1, d1test], sort=False)

d1.isnull().sum()

# Drop unwanted columns
col_to_drop = ["recommendation_set_id", "user_id", "session_id", "user_os_version", "user_java_version",
               "time_recs_recieved", "time_recs_displayed", "time_recs_viewed", "user_os", "query_identifier",
               "document_language_provided", "user_timezone", "timezone_by_ip", "response_delivered", 
               "number_of_recs_in_set", "application_type", "item_type", "organization_id", "rec_processing_time",
               "ctr", "clicks"]
d1.drop(col_to_drop, axis=1, inplace=True)

# Initialise label encoder 
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
scaler = preprocessing.MinMaxScaler()

# Preprocess each column
d1["query_detected_language"].fillna(d1["query_detected_language"].mode()[0], inplace=True )
d1['query_detected_language']= label_encoder.fit_transform(d1['query_detected_language'])

d1["query_document_id"].fillna(d1["query_document_id"].mode()[0], inplace=True )
d1['query_document_id'] = scaler.fit_transform(d1['query_document_id'].values.reshape(-1,1))

d1["year_published"].fillna(d1["year_published"].mode()[0], inplace=True )

d1["number_of_authors"].fillna(d1["number_of_authors"].mode()[0], inplace=True )
d1['number_of_authors'] = scaler.fit_transform(d1['number_of_authors'].values.reshape(-1,1))

d1["query_char_count"].fillna(d1["query_char_count"].mode()[0], inplace=True )
d1['query_char_count'] = scaler.fit_transform(d1['query_char_count'].values.reshape(-1,1))

d1["abstract_char_count"].fillna(d1["abstract_char_count"].mode()[0], inplace=True )
d1['abstract_char_count'] = scaler.fit_transform(d1['abstract_char_count'].values.reshape(-1,1))

d1["abstract_word_count"].fillna(d1["abstract_word_count"].mode()[0], inplace=True )
d1['abstract_word_count'] = scaler.fit_transform(d1['abstract_word_count'].values.reshape(-1,1))

d1["abstract_detected_language"].fillna(d1["abstract_detected_language"].mode()[0], inplace=True )
d1['abstract_detected_language']= label_encoder.fit_transform(d1['abstract_detected_language'])

d1["first_author_id"].fillna(d1["first_author_id"].mode()[0], inplace=True )
d1['first_author_id']= scaler.fit_transform(d1['first_author_id'].values.reshape(-1,1))

d1["num_pubs_by_first_author"].fillna(d1["num_pubs_by_first_author"].mode()[0], inplace=True )
d1['num_pubs_by_first_author']= label_encoder.fit_transform(d1['num_pubs_by_first_author'])

d1["request_received"].fillna(d1["request_received"].mode()[0], inplace=True )
d1['request_received']= label_encoder.fit_transform(d1['request_received'])
d1['request_received']= scaler.fit_transform(d1['request_received'].values.reshape(-1,1))

#d1["response_delivered"] = calc_smooth_mean(d1, 'response_delivered', 'set_clicked', 50)
#d1["response_delivered"].fillna(d1["response_delivered"].mode()[0], inplace=True )

d1["app_version"].fillna(d1["app_version"].mode()[0], inplace=True )
d1['app_version']= label_encoder.fit_transform(d1['app_version'])

d1["app_lang"].fillna(d1["app_lang"].mode()[0], inplace=True )
d1['app_lang']= label_encoder.fit_transform(d1['app_lang'])

d1["country_by_ip"].fillna(d1["country_by_ip"].mode()[0], inplace=True )
d1['country_by_ip']= label_encoder.fit_transform(d1['country_by_ip'])

d1["local_time_of_request"].fillna(d1["local_time_of_request"].mode()[0], inplace=True )
d1['local_time_of_request']= label_encoder.fit_transform(d1['local_time_of_request'])
d1['local_time_of_request']= scaler.fit_transform(d1['local_time_of_request'].values.reshape(-1,1))

d1["local_hour_of_request"].fillna(d1["local_hour_of_request"].mode()[0], inplace=True )
d1['local_hour_of_request']= label_encoder.fit_transform(d1['local_hour_of_request'])

d1["recommendation_algorithm_id_used"].fillna(d1["recommendation_algorithm_id_used"].mode()[0], inplace=True )

d1['algorithm_class']= label_encoder.fit_transform(d1['algorithm_class'])

d1["cbf_parser"] = d1["cbf_parser"].replace(pd.np.nan, "no") 
d1['cbf_parser']= label_encoder.fit_transform(d1['cbf_parser'])

d1['search_title']= label_encoder.fit_transform(d1['search_title'])

d1['search_abstract']= label_encoder.fit_transform(d1['search_abstract'])

d1['search_keywords']= label_encoder.fit_transform(d1['search_keywords'])

d1['query_word_count']= scaler.fit_transform(d1['query_word_count'].values.reshape(-1,1))
d1['query_detected_language']= scaler.fit_transform(d1['query_detected_language'].values.reshape(-1,1))
d1['year_published']= scaler.fit_transform(d1['year_published'].values.reshape(-1,1))
d1['abstract_detected_language']= scaler.fit_transform(d1['abstract_detected_language'].values.reshape(-1,1))
d1['num_pubs_by_first_author']= scaler.fit_transform(d1['num_pubs_by_first_author'].values.reshape(-1,1))
d1['hour_request_received']= scaler.fit_transform(d1['hour_request_received'].values.reshape(-1,1))
d1['app_version']= scaler.fit_transform(d1['app_version'].values.reshape(-1,1))
d1['app_lang']= scaler.fit_transform(d1['app_lang'].values.reshape(-1,1))
d1['country_by_ip']= scaler.fit_transform(d1['country_by_ip'].values.reshape(-1,1))
d1['local_hour_of_request']= scaler.fit_transform(d1['local_hour_of_request'].values.reshape(-1,1))
d1['recommendation_algorithm_id_used']= scaler.fit_transform(d1['recommendation_algorithm_id_used'].values.reshape(-1,1))
d1['algorithm_class']= scaler.fit_transform(d1['algorithm_class'].values.reshape(-1,1))
d1['cbf_parser']= scaler.fit_transform(d1['cbf_parser'].values.reshape(-1,1))

# keep a copy, incase something happens after this!!!
d1copy = d1.copy()

# Split back Train data
x = d1[d1["isTrain"]==1]
y_full = x['set_clicked']
x_full = x.drop(columns=['set_clicked','isTrain'])

# Split train and holdout data (80-20)
from sklearn.model_selection import train_test_split
x_train, x_holdOut, y_train, y_holdOut = train_test_split(x_full, y_full, train_size=0.8, random_state=10)

# Split back Test data
x_t = d1[d1["isTrain"]==0]
x_test = x_t.drop(columns=['set_clicked','isTrain'])


# Train model on various parameter values - identify best performing parameter values
n_estimators = [1500, 2000]
#depth = [3, 4, 5, 6, 7, 8, 9, 10]
for iters in n_estimators:
    model=RandomForestClassifier(n_estimators=iters, random_state=100)
    model.fit(x_train, y_train)  #,plot=True)
    ypred = model.predict(x_holdOut)
    print ("---------------------------Iterations    : " + str(iters))
    print ("---------------------------F1 Score      : " + str(f1_score(y_holdOut, ypred)))
    print ("-----------------------------------------------------------------------")

# Train model with parameter values identified above
model=RandomForestClassifier(n_estimators=1500, random_state=100)
model.fit(x_full, y_full)  #,plot=True)

# Final predict on test data
final_pred = model.predict(x_test)
pd.DataFrame(final_pred).to_csv("E:/Trinity/Machine Learning/Kaggle/Project 2/tcd-ml-comp-201920-rec-alg-click-pred-group/d1.csv")
