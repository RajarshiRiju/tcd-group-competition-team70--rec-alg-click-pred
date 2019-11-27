#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
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

# Filter only Org_id 4 data
d4=dataset[dataset["organization_id"]==4]
d4test=datatest[datatest["organization_id"]==4] 
#len_train = len(d4)
#len_test  = len(d4test)

# Add a column to differentiate test and train data
d4["isTrain"] = 1
d4test["isTrain"] = 0

# Concat Test and Train datasets
d4 = pd.concat([d4, d4test], sort=False)


d4.isnull().sum()

# Drop unwanted columns
col_to_drop = ["recommendation_set_id", "session_id", "user_id", "document_language_provided", "year_published", 
               "number_of_authors", "first_author_id", "num_pubs_by_first_author", "app_version", "user_os",
               "user_os_version", "user_java_version", "rec_processing_time", "organization_id", "user_timezone",
               "timezone_by_ip", "time_recs_recieved", "time_recs_displayed", "time_recs_viewed", "number_of_recs_in_set", 
               "application_type", "response_delivered", "ctr", "clicks"]

d4.drop(col_to_drop, axis=1, inplace=True)

# Initialise label encoder 
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
scaler = preprocessing.MinMaxScaler()

# Preprocess each column
d4["query_detected_language"].fillna(d4["query_detected_language"].mode()[0], inplace=True )
d4['query_detected_language']= label_encoder.fit_transform(d4['query_detected_language'])

d4["query_word_count"].fillna(d4["query_word_count"].mode()[0], inplace=True )
d4["query_word_count"] = d4["query_word_count"].astype(int)
d4['query_word_count'] = scaler.fit_transform(d4['query_word_count'].values.reshape(-1,1))

d4["query_document_id"].fillna(d4["query_document_id"].mode()[0], inplace=True )
d4['query_document_id'] = scaler.fit_transform(d4['query_document_id'].values.reshape(-1,1))

d4["query_identifier"] = d4["query_identifier"].str.split("|", expand=True,)[0]
d4["query_identifier"].fillna(d4["query_identifier"].mode()[0], inplace=True )
d4['query_identifier']= label_encoder.fit_transform(d4['query_identifier'])

#d1["abstract_word_count"].fillna(d1["abstract_word_count"].mode()[0], inplace=True )
#d1['abstract_word_count'] = scaler.fit_transform(d1['abstract_word_count'].values.reshape(-1,1))

d4["query_char_count"].fillna(d4["query_char_count"].mode()[0], inplace=True )
d4['query_char_count'] = scaler.fit_transform(d4['query_char_count'].values.reshape(-1,1))

d4["abstract_char_count"].fillna(d4["abstract_char_count"].mode()[0], inplace=True )
d4['abstract_char_count'] = scaler.fit_transform(d4['abstract_char_count'].values.reshape(-1,1))

d4["abstract_word_count"].fillna(d4["abstract_word_count"].mode()[0], inplace=True )
d4['abstract_word_count'] = scaler.fit_transform(d4['abstract_word_count'].values.reshape(-1,1))

d4["abstract_detected_language"].fillna(d4["abstract_detected_language"].mode()[0], inplace=True )
d4['abstract_detected_language']= label_encoder.fit_transform(d4['abstract_detected_language'])

d4["request_received"].fillna(d4["request_received"].mode()[0], inplace=True )
d4['request_received']= label_encoder.fit_transform(d4['request_received'])
d4['request_received']= scaler.fit_transform(d4['request_received'].values.reshape(-1,1))

d4["item_type"].fillna(d4["item_type"].mode()[0], inplace=True )
d4['item_type']= label_encoder.fit_transform(d4['item_type'])

d4["app_lang"].fillna(d4["app_lang"].mode()[0], inplace=True )
d4['app_lang']= label_encoder.fit_transform(d4['app_lang'])

d4["country_by_ip"].fillna(d4["country_by_ip"].mode()[0], inplace=True )
d4['country_by_ip']= label_encoder.fit_transform(d4['country_by_ip'])

d4["local_time_of_request"].fillna(d4["local_time_of_request"].mode()[0], inplace=True )
d4['local_time_of_request']= label_encoder.fit_transform(d4['local_time_of_request'])
d4['local_time_of_request']= scaler.fit_transform(d4['local_time_of_request'].values.reshape(-1,1))

d4["local_hour_of_request"].fillna(d4["local_hour_of_request"].mode()[0], inplace=True )
d4['local_hour_of_request']= label_encoder.fit_transform(d4['local_hour_of_request'])

#d4["recommendation_algorithm_id_used"] = calc_smooth_mean(d4, 'recommendation_algorithm_id_used', 'set_clicked', 50)
d4["recommendation_algorithm_id_used"].fillna(d4["recommendation_algorithm_id_used"].mode()[0], inplace=True )

d4['algorithm_class']= label_encoder.fit_transform(d4['algorithm_class'])

d4["cbf_parser"] = d4["cbf_parser"].replace(pd.np.nan, "no") 
d4['cbf_parser']= label_encoder.fit_transform(d4['cbf_parser'])

d4['search_title']= label_encoder.fit_transform(d4['search_title'])

d4['search_abstract']= label_encoder.fit_transform(d4['search_abstract'])

d4['search_keywords']= label_encoder.fit_transform(d4['search_keywords'])

d4['query_detected_language']= scaler.fit_transform(d4['query_detected_language'].values.reshape(-1,1))
d4['abstract_detected_language']= scaler.fit_transform(d4['abstract_detected_language'].values.reshape(-1,1))
d4['hour_request_received']= scaler.fit_transform(d4['hour_request_received'].values.reshape(-1,1))
d4['app_lang']= scaler.fit_transform(d4['app_lang'].values.reshape(-1,1))
d4['country_by_ip']= scaler.fit_transform(d4['country_by_ip'].values.reshape(-1,1))
d4['local_hour_of_request']= scaler.fit_transform(d4['local_hour_of_request'].values.reshape(-1,1))
d4['recommendation_algorithm_id_used']= scaler.fit_transform(d4['recommendation_algorithm_id_used'].values.reshape(-1,1))
d4['algorithm_class']= scaler.fit_transform(d4['algorithm_class'].values.reshape(-1,1))
d4['cbf_parser']= scaler.fit_transform(d4['cbf_parser'].values.reshape(-1,1))
d4['query_identifier']= scaler.fit_transform(d4['query_identifier'].values.reshape(-1,1))
d4['item_type']= scaler.fit_transform(d4['item_type'].values.reshape(-1,1))

# keep a copy, incase something happens after this!!!
d4copy = d4.copy()

# Split back Train data
x = d4[d4["isTrain"]==1]
y_full = x['set_clicked']
x_full = x.drop(columns=['set_clicked','isTrain'])

# Split train and holdout data (80-20)
from sklearn.model_selection import train_test_split
x_train, x_holdOut, y_train, y_holdOut = train_test_split(x_full, y_full, train_size=0.8, random_state=10)

# Split back Test data
x_t = d4[d4["isTrain"]==0]
x_test = x_t.drop(columns=['set_clicked','isTrain'])


# Train model on various parameter values - identify best performing parameter values
iterations = [10, 50, 100, 200, 500]
learning_rate = [0.01, 0.02, 0.05, 0.075, 0.1, 0.2, 0.5, 0.75, 1.0]
depth = [3, 4, 5, 6, 7, 8, 9, 10]

for iters in iterations:        #bad code - nested for!
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
# ---------------------------Learning Rate : 0.5
# ---------------------------Depth         : 11
# ---------------------------F1 Score      : 0.9358372456964006

# Train model with parameter values identified above
from sklearn.metrics import classification_report, confusion_matrix, f1_score
model=CatBoostClassifier(iterations=500, depth=5, learning_rate=0.5, verbose=False)
model.fit(x_full, y_full)  

# Final predict on test data
final_pred = model.predict(x_test)
pd.DataFrame(final_pred).to_csv("E:/Trinity/Machine Learning/Kaggle/Project 2/tcd-ml-comp-201920-rec-alg-click-pred-group/d4.csv")

