# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 08:03:14 2021

@author: Richa
"""

import os
os.chdir("D:\\Study\\Marko Mine\\Flowrate")

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report

station = 'FRO_KC1_filtered'
flowrate = pd.read_csv(station+'_.csv', usecols=[2, 3])

avg_days_DT = 1#here is the average days for decision tree input
flowrate_threshold = 1.2#1.2, 1.7, and 0.7 for FRO_KC1, FRO_HC1, and EVO_HC1
seed = 1029

train_startDate = '1990-01-01'
test_startDate = '2013-01-01'
endDate = '2013-12-31'

flowrate.columns = ['sample_date', 'flow']
# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)

#print(flowrate.describe())

# =============================================================================
# Missing weather data filling
# =============================================================================
try:
    #2 choose 1
    weather = pd.read_csv('Weather_filled_avg_' + str(avg_days_DT) + '.csv').drop('Num', 1)#weather data for the tree must NOT be averaged
    #weather = pd.read_csv('Weather_long_filled_avg_' + str(avg_days_DT) + '.csv').drop('Num', 1)#weather data for the tree must NOT be averaged
    
    weather['Datetime'] = pd.to_datetime(weather['Datetime'], format='%Y/%m/%d')
    print('Filled weather data loaded successfully')
except:
    print('Filled weather data not detected, generating...')
    #weather = pd.read_csv('en_climate_daily_BC_1157630_1990-2013_P1D.csv', 
    #                  usecols=[4, 5, 6, 7, 13, 19, 21, 23, 25]) 
    #2 choose 1
    weather = pd.read_csv('weather_1990-2013_avg_' + str(avg_days_DT) + '.csv')
    #weather = pd.read_csv('weather_1980-2020_avg_' + str(avg_days_DT) + '.csv')
    
    weather['Datetime'] = pd.to_datetime(weather['Date/Time'], format='%Y/%m/%d')
    weather = weather.drop('Date/Time', 1)
    monthly_mean = pd.DataFrame()
    monthly_mean['Mean Temp (C)_1'] = weather.groupby('Month')['Mean Temp (C)'].mean()
    monthly_mean['Total Rain (mm)_1'] = weather.groupby('Month')['Total Rain (mm)'].mean()
    monthly_mean['Total Snow (cm)_1'] = weather.groupby('Month')['Total Snow (cm)'].mean()
    monthly_mean['Total Precip (mm)_1'] = weather.groupby('Month')['Total Precip (mm)'].mean()
    monthly_mean['Snow on Grnd (cm)_1'] = weather.groupby('Month')['Snow on Grnd (cm)'].mean()
    monthly_mean['Month'] = monthly_mean.index
    
    weather_copy = weather.copy()
    monthly_mean.index = range(1, 13)#for 12 months, the name of index must be removed
    weather_copy = pd.merge(weather, monthly_mean, on=('Month'), how='left')

    for i in ['Mean Temp (C)', 'Total Rain (mm)', 'Total Snow (cm)', 'Total Precip (mm)', 'Snow on Grnd (cm)']:
        for j in range(0, len(weather_copy)):
            if weather_copy[i].isnull()[j]:
                weather_copy.loc[j, i] = weather_copy[i+'_1'][j]
    weather = weather_copy.drop(columns=['Mean Temp (C)_1', 'Total Rain (mm)_1', 
                                         'Total Snow (cm)_1', 'Total Precip (mm)_1',
                                         'Snow on Grnd (cm)_1'])
    #2 choose 1
    pd.DataFrame(weather).to_csv('Weather_filled_avg_' + str(avg_days_DT) + '.csv')
    #pd.DataFrame(weather).to_csv('Weather_long_filled_avg_' + str(avg_days_DT) + '.csv')
    
    print('Filled weather data saved successfully')
#print(weather.describe())
# =============================================================================
# Generating melting data
# =============================================================================
melt = np.zeros(len(flowrate['flow']))
j = 0
for i in flowrate['flow']:
    if i > flowrate_threshold:#Spring freshet
       melt[j]  = 1
    else:
        melt[j] = 0
    j = j + 1

flowrate['melt'] = melt

merge = pd.merge(weather, flowrate, on=('Datetime'), how='left').drop('flow', 1)
merge = np.array(merge)

day0 = []
#day_1, day_2 = [], []#for more than 2 days
for i in range(0, len(merge)):
    if merge[i][9] == 0 or merge[i][9] == 1:
        day0.append(merge[i, :])
        #day_1.append(merge[i-1, 3:8])#start/end columns are to be changed, check by hand
        #day_2.append(merge[i-2, 3:8])
day0 = np.array(day0)
#day_1, day_2 = np.array(day_1), np.array(day_2)#for more than 2 days

# Switch of 1 day, 2 days or 3 days
X = day0.copy()
#X = np.c_[day0, day_1]
#X = np.c_[day0, day_1, day_2]

# Transfer to dataframe and seperate to train, valid and test set
X = pd.DataFrame(X, index=X[:, 8])
X = X.drop(2,1)
X.dropna(inplace=True)#just for double-check there won't be nan to feed the DT

X_test = X.loc[test_startDate : endDate].values
X = X.loc[train_startDate : test_startDate].values#Changed
#datetime = X[:, 7]
y = X[:, 8]

X = X[:, 1:6]
#X = np.c_[X[:, 1:7], X[:, 9:]]#for more than 2 days
#eliminate 'year' from the inputX = X[:, 1:8] 

# =============================================================================
# Transforming test set
# =============================================================================
y_test = X_test[:, 8]

X_test = X_test[:, 1:6]
#X_test = np.c_[X_test[:, 1:7], X_test[:, 9:]]#for more than 2 days
#X = X[:, 1:8] eliminate 'year' from the input

# =============================================================================
# Defining functions
# =============================================================================
def predict_test(X_scaled_test, classifier):
    y_pred = classifier.predict(X_scaled_test)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), (y_test.reshape(len(y_test), 1))), 1))
    return y_pred

def accuracy_print_conf(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    conf_matrix = confusion_matrix(y_test.astype('int'), y_pred.astype('int'))
    print('The confusion matrix is:\n', conf_matrix)
    accuracy = accuracy_score(y_test.astype('int'), y_pred.astype('int'))
    print('The accuracy is: %2.2f' % accuracy)
    return accuracy

def rootMSE(y_test, y_pred):
    import math
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE = %2.2f' % rmse)
    print('Predicted results length:', y_pred.shape)
    y_test = np.array(y_test).reshape(-1, 1)
    print('Real results length:', y_test.shape)
    return rmse

# Fixing a seed
np.random.seed(seed)

# =============================================================================
# Random Forest
# =============================================================================

try:
    from joblib import load
    classifier = load('RFForLSTM_new'+station+'.joblib')
    print('Trained decision tree result loaded successfully')
except:
    print('No training result detected, training...')
    from sklearn.ensemble import RandomForestClassifier
    #feature_name=[weather.columns[1], weather.columns[3], weather.columns[4], 
    #              weather.columns[5], weather.columns[6], weather.columns[7]]
    classifier = RandomForestClassifier()
    # Grid searching
    #from sklearn.model_selection import GridSearchCV
    parameters = {'criterion':('gini', 'entropy'),
                  'n_estimators':(50, 100, 10),
                  'min_weight_fraction_leaf':(0.1, 0.01)}
    clf = GridSearchCV(classifier, parameters,n_jobs=-1, cv=5)
    clf.fit(X, y.astype('int'))
    print('Best score:', clf.best_score_)
    print('Best parameters:', clf.best_params_)
    
    # Training the classifier with best hyperparameters
    classifier = RandomForestClassifier(criterion=clf.best_params_.get('criterion'),
                                        n_estimators=clf.best_params_.get('n_estimators'),
                                        min_weight_fraction_leaf=clf.best_params_.get('min_weight_fraction_leaf'))
    classifier.fit(X, y.astype('int'))
    
    from joblib import dump
    dump(classifier, 'RFForLSTM_new'+station+'.joblib')#To be changed
    print('Decision tree training result saved')
'''
#fixed hps
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = 'gini',
                                    n_estimators = 10,
                                    min_weight_fraction_leaf = 0.1)
classifier.fit(X, y.astype('int'))
'''
# Prediction
print('For Random Forest Classifier: ')
y_pred = predict_test(X_test, classifier)
accuracy = accuracy_print_conf(y_test, y_pred)

#Indicator calculating
dt_roc_auc = roc_auc_score(np.int32(y_test), y_pred)
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(np.int32(y_test), y_pred))
'''
The confusion matrix is:
 [[12  2]
 [ 0  9]]
The accuracy is: 0.91
Decision Tree AUC = 0.93
              precision    recall  f1-score   support

           0       1.00      0.86      0.92        14
           1       0.82      1.00      0.90         9

    accuracy                           0.91        23
   macro avg       0.91      0.93      0.91        23
weighted avg       0.93      0.91      0.91        23
'''

# =============================================================================
# XGBoosting
# =============================================================================
'''
try:
    from joblib import load
    classifier = load('XGBForLSTM_new'+station+'.joblib')
    print('Trained decision tree result loaded successfully')
except:
    print('No training result detected, training...')
    import xgboost as xgb
    #feature_name=[weather.columns[1], weather.columns[3], weather.columns[4], 
    #              weather.columns[5], weather.columns[6], weather.columns[7]]
    #classifier = xgb.XGBClassifier(max_depth=5, n_estimators=10, silent=True, objective='binary:hinge', feature_names=feature_name)
    classifier = xgb.XGBClassifier(max_depth=5, n_estimators=10, silent=True, objective='binary:hinge')
    classifier.fit(X, y.astype('int'))
    # Grid searching
    #from sklearn.model_selection import GridSearchCV
    parameters = {'n_estimators':(5, 10, 50),
                  'max_depth':(3, 5, 9)}
    clf = GridSearchCV(classifier, parameters,n_jobs=-1, cv=5)
    clf.fit(X, y.astype('int'))
    print('Best score:', clf.best_score_)
    print('Best parameters:', clf.best_params_)
    
    # Training the classifier with best hyperparameters
    classifier = xgb.XGBClassifier(max_depth=clf.best_params_.get('max_depth'),
                                   n_estimators=clf.best_params_.get('n_estimators'),
                                   silent=True,
                                   objective='binary:hinge', #feature_names=feature_name, 
                                   random_state=seed, use_label_encoder =False)
    classifier.fit(X, y.astype('int'))
    
    from joblib import dump
    dump(classifier, ''XGBForLSTM_new'+station+'.joblib')#To be changed
    print('Decision tree training result saved')
'''
#fixed hps
import xgboost as xgb
classifier = xgb.XGBClassifier(max_depth=3,
                               n_estimators=50,
                               #silent=True,
                               objective='binary:hinge', #feature_names=feature_name, 
                               random_state=seed, use_label_encoder =False)
classifier.fit(X, y.astype('int'))

# Prediction
print('For XGBoosting Classifier: ')
y_pred = predict_test(X_test, classifier)
accuracy = accuracy_print_conf(y_test, y_pred)

#Indicator calculating
dt_roc_auc = roc_auc_score(np.int32(y_test), y_pred)
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(np.int32(y_test), y_pred))

'''
Day 0:
The confusion matrix is:
 [[12  2]
 [ 1  8]]
The accuracy is: 0.87
Decision Tree AUC = 0.87
              precision    recall  f1-score   support

           0       0.92      0.86      0.89        14
           1       0.80      0.89      0.84         9

    accuracy                           0.87        23
   macro avg       0.86      0.87      0.87        23
weighted avg       0.87      0.87      0.87        23
'''

# =============================================================================
# AdaBoosting
# =============================================================================
'''
try:
    from joblib import load
    classifier = load('AdaBForLSTM_new'+station+'.joblib')
    print('Trained decision tree result loaded successfully')
except:
    print('No training result detected, training...')
    from sklearn.ensemble import AdaBoostClassifier
    #feature_name=[weather.columns[1], weather.columns[3], weather.columns[4], 
    #              weather.columns[5], weather.columns[6], weather.columns[7]]
    classifier = AdaBoostClassifier(algorithm='SAMME',
                   base_estimator=DecisionTreeClassifier(criterion='gini',
                                                         min_weight_fraction_leaf=0.1),
                   learning_rate=0.8, n_estimators=300, random_state=seed)
    # Grid searching
    #from sklearn.model_selection import GridSearchCV
    parameters = {'learning_rate':(0.08, 0.8),
                  'n_estimators':(50, 100, 10)}
    clf = GridSearchCV(classifier, parameters,n_jobs=-1, cv=5)
    clf.fit(X, y.astype('int'))
    print('Best score:', clf.best_score_)
    print('Best parameters:', clf.best_params_)
    
    # Training the classifier with best hyperparameters
    classifier = AdaBoostClassifier(algorithm='SAMME',
                                    base_estimator=DecisionTreeClassifier(criterion='gini',
                                                         min_weight_fraction_leaf=0.1),
                                    learning_rate=clf.best_params_.get('learning_rate'),
                                    n_estimators=clf.best_params_.get('n_estimators'),
                                   random_state=seed)
    classifier.fit(X, y.astype('int'))
    
    from joblib import dump
    dump(classifier, 'AdaBForLSTM_new'+station+'.joblib')#To be changed
    print('Decision tree training result saved')
'''
#fixed hps
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(algorithm='SAMME.R',
                                base_estimator=DecisionTreeClassifier(criterion='gini',
                                                     min_weight_fraction_leaf=0.1),
                                learning_rate = 0.08,
                                n_estimators = 10,
                                random_state=seed)
classifier.fit(X, y.astype('int'))
# Prediction
print('For AdaBoosting Classifier: ')
y_pred = predict_test(X_test, classifier)
accuracy = accuracy_print_conf(y_test, y_pred)

#Indicator calculating
dt_roc_auc = roc_auc_score(np.int32(y_test), y_pred)
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(np.int32(y_test), y_pred))

'''
Day 0:
Best score: 0.8576491228070175
Best parameters: {'learning_rate': 0.08, 'n_estimators': 10}
The confusion matrix is:
 [[12  2]
 [ 0  9]]
The accuracy is: 0.91
Decision Tree AUC = 0.93
              precision    recall  f1-score   support

           0       1.00      0.86      0.92        14
           1       0.82      1.00      0.90         9

    accuracy                           0.91        23
   macro avg       0.91      0.93      0.91        23
weighted avg       0.93      0.91      0.91        23
'''

# =============================================================================
# Decision Tree
# =============================================================================
try:
    from joblib import load
    classifier = load('DecisionTreeForLSTM_new'+station+'.joblib')
    print('Trained decision tree result loaded successfully')
except:
    print('No training result detected, training...')
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()# Use 'entropy' instead of 'gini'
    
    # Grid searching
    parameters = {'criterion':('gini', 'entropy'),
                  'min_weight_fraction_leaf':(0.1, 0.01)}
    clf = GridSearchCV(classifier, parameters,n_jobs=-1, cv=5)
    clf.fit(X, y.astype('int'))
    print('Best score:', clf.best_score_)
    print('Best parameters:', clf.best_params_)
    
    # Training the classifier with best hyperparameters
    classifier = DecisionTreeClassifier(criterion=clf.best_params_.get('criterion'),
                                        min_weight_fraction_leaf=clf.best_params_.get('min_weight_fraction_leaf'),
                                        random_state=seed)
    classifier.fit(X, y.astype('int'))
    
    from joblib import dump
    dump(classifier, 'DecisionTreeForLSTM_new'+station+'.joblib')#To be changed
    print('Decision tree training result saved')

# Prediction
print('For DecisionTreeClassifier: ')
y_pred = predict_test(X_test, classifier)
accuracy = accuracy_print_conf(y_test, y_pred)

#Indicator calculating
try:
    from sklearn.metrics import roc_auc_score, classification_report
    dt_roc_auc = roc_auc_score(np.int32(y_test), y_pred)
    print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
    print(classification_report(np.int32(y_test), y_pred))
except:
    print("ROC doesn't exist")

'''
The confusion matrix is:
 [[14  0]
 [ 2  7]]
The accuracy is: 0.91
Decision Tree AUC = 0.89
              precision    recall  f1-score   support

           0       0.88      1.00      0.93        14
           1       1.00      0.78      0.88         9

    accuracy                           0.91        23
   macro avg       0.94      0.89      0.90        23
weighted avg       0.92      0.91      0.91        23
'''
