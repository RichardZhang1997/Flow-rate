# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 18:18:33 2020

@author: Richa
"""

import os
os.chdir("C:\\MyFile\\Study\\Graduate\\Marko MIne\\Flowrate")

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Loading datasets
flowrate = pd.read_csv('FRO_KC1_.csv', usecols=[2, 10])
weather = pd.read_csv('en_climate_daily_BC_1157630_1990-2013_P1D.csv', 
                      usecols=[4, 5, 6, 7, 13, 19, 21, 23, 25]) 

# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
weather['Datetime'] = pd.to_datetime(weather['Date/Time'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)
weather = weather.drop('Date/Time', 1)

'''
# =============================================================================
# EDA and Visualization
# =============================================================================
# Year bar chart
weather.groupby('Year')['Mean Temp (°C)'].mean().drop(index=[2008,2009]).plot.bar(title='Average Mean Temperature (°C) by Year')

weather.groupby('Year')['Total Rain (mm)'].mean().drop(index=[2008,2009]).plot.bar(title='Average Total Rain (mm) by Year')

weather.groupby('Year')['Total Snow (cm)'].mean().drop(index=[2008,2009]).plot.bar(title='Average Total Snow (cm) by Year')

weather.groupby('Year')['Total Precip (mm)'].mean().drop(index=[2008,2009]).plot.bar(title='Average Total Precip (mm) by Year')

weather.groupby('Year')['Snow on Grnd (cm)'].mean().drop(index=[2008,2009]).plot.bar(title='Average Snow on Grnd (cm) by Year')

# Month bar chart
weather.groupby('Month')['Mean Temp (°C)'].mean().plot.bar(title='Average Mean Temperature (°C) by Month')

weather.groupby('Month')['Total Rain (mm)'].mean().plot.bar(title='Average Total Rain (mm) by Month')

weather.groupby('Month')['Total Snow (cm)'].mean().plot.bar(title='Average Total Snow (cm) by Month')

weather.groupby('Month')['Total Precip (mm)'].mean().plot.bar(title='Average Total Precip (mm) by Month')

weather.groupby('Month')['Snow on Grnd (cm)'].mean().plot.bar(title='Average Snow on Grnd (cm) by Month')

# Month pie chart
explode = [0, 0, 0, 0, 0.1, 0.1, 0, 0, 0, 0, 0.15, 0 ]#数字越大突出显示越大
colors = ['red', 'blue', 'yellow', 'green', 'darkred', 'purple', 'cyan', 'brown', 'orange', 'gray', 'pink', 'crimson']

temp = weather.groupby('Month')['Total Rain (mm)'].mean()#.plot.pie(title='Average Total Rain (mm) by Month')
plt.pie(x=temp, explode=explode, labels=temp.index, colors=colors,
        autopct='%.1f%%', pctdistance=0.8, labeldistance=1.1, startangle=120,
        radius=1.2, counterclock=False,
        wedgeprops={'linewidth':1.5,'edgecolor':'black'},
        textprops={'fontsize':8, 'color':'black'})#保留一位小数
plt.title('Average Total Rain (mm) by Month')
plt.show()

temp = weather.groupby('Month')['Total Precip (mm)'].mean()#.plot.pie(title='Average Total Precip (mm) by Month')
plt.pie(x=temp, explode=explode, labels=temp.index, colors=colors,
        autopct='%.1f%%', pctdistance=0.8, labeldistance=1.1, startangle=120,
        radius=1.2, counterclock=False,
        wedgeprops={'linewidth':1.5,'edgecolor':'black'},
        textprops={'fontsize':8, 'color':'black'})#保留一位小数
plt.title('Average Total Precip (mm) by Month')
plt.show()
'''

# =============================================================================
# Missing weather data filling
# =============================================================================
'''
monthly_mean = pd.DataFrame()
monthly_mean['Mean Temp (°C)_1'] = weather.groupby('Month')['Mean Temp (°C)'].mean()
monthly_mean['Total Rain (mm)_1'] = weather.groupby('Month')['Total Rain (mm)'].mean()
monthly_mean['Total Snow (cm)_1'] = weather.groupby('Month')['Total Snow (cm)'].mean()
monthly_mean['Total Precip (mm)_1'] = weather.groupby('Month')['Total Precip (mm)'].mean()
monthly_mean['Snow on Grnd (cm)_1'] = weather.groupby('Month')['Snow on Grnd (cm)'].mean()
monthly_mean['Month'] = monthly_mean.index

weather_copy = weather.copy()
monthly_mean.index = range(1, 13)#for 12 months, the name of index must be removed
weather_copy = pd.merge(weather, monthly_mean, on=('Month'), how='left')

for i in ['Mean Temp (°C)', 'Total Rain (mm)', 'Total Snow (cm)', 'Total Precip (mm)', 'Snow on Grnd (cm)']:
    for j in range(0, len(weather_copy)):
        if weather_copy[i].isnull()[j]:
            weather_copy.loc[j, i] = weather_copy[i+'_1'][j]
weather = weather_copy.drop(columns=['Mean Temp (°C)_1', 'Total Rain (mm)_1', 
                                     'Total Snow (cm)_1', 'Total Precip (mm)_1',
                                     'Snow on Grnd (cm)_1'])
'''
#pd.DataFrame(weather).to_csv('Weather_filled.csv')
weather = pd.read_csv('Weather_filled.csv').drop('Num', 1)
weather['Datetime'] = pd.to_datetime(weather['Datetime'], format='%Y/%m/%d')
# =============================================================================
# Generating melting data
# =============================================================================
flowrate_threshold = 2
melt = np.zeros(len(flowrate['flow']))
j = 0
for i in flowrate['flow']:
    if i > flowrate_threshold:
       melt[j]  = 1
    else:
        melt[j] = 0
    j = j + 1

flowrate['melt'] = melt

merge = pd.merge(weather, flowrate, on=('Datetime'), how='left').drop('flow', 1)
merge = np.array(merge)

day0, day_1, day_2 = [], [], []
refering_day = 3
for i in range(0, len(merge)):
    if merge[i][9] == 0 or merge[i][9] == 1:
        day0.append(merge[i, :])
        day_1.append(merge[i-1, 3:8])#start/end columns are to be changed, check by hand
        day_2.append(merge[i-2, 3:8])
day0 ,day_1, day_2 = np.array(day0), np.array(day_1), np.array(day_2)

# Switch of 1day, 2days or 3days
#X = day0.copy()
X = np.c_[day0, day_1]
#X = np.c_[day0, day_1, day_2]

# Transfer to dataframe and seperate to train, valid and test set
X = pd.DataFrame(X, index=X[:, 8])
X.dropna(inplace=True)

# One_Hot Incoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = pd.DataFrame(X, index=X[:, 19])

test = X.loc['2013-01-01':'2013-12-31'].values
X = X.loc['1992-01-01':'2013-01-01'].values#Changed
datetime = X[:, 19]
y = X[:, 20]
X = np.c_[X[:, :19], X[:, 21:]]

# Seperating results of One_Hot encoding
X_month = X[:, :12]
X = X[:, 12:]

# Feature scaling for both X and y
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_scaled = sc_X.fit_transform(X)

X_scaled = np.c_[X_month, X_scaled]

# =============================================================================
# Transforming test set
# =============================================================================
datetime_test = test[:, 19]
y_test = test[:, 20]
test = np.c_[test[:, :19], test[:, 21:]]

# Seperating results of One_Hot encoding
X_month_test = test[:, :12]
test = test[:, 12:]

X_scaled_test = sc_X.transform(test)
X_scaled_test = np.c_[X_month_test, X_scaled_test]

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
    print('The accuracy is:', accuracy)
    return accuracy

def rootMSE(y_test, y_pred):
    import math
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE =', rmse)
    return rmse

# =============================================================================
# Decision Tree
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()# Use 'entropy' instead of 'gini'

# Grid searching
from sklearn.model_selection import GridSearchCV
parameters = {'criterion':('gini', 'entropy')}
clf = GridSearchCV(classifier, parameters,n_jobs=-1, cv=5)
clf.fit(X_scaled, y.astype('int'))
print('Best score:', clf.best_score_)
print('Best parameters:', clf.best_params_)

# Training the classifier with best hyperparameters
classifier = DecisionTreeClassifier(criterion=clf.best_params_.get('criterion'),
                                    random_state=1029)
classifier.fit(X_scaled, y.astype('int'))

# Prediction
print('For DecisionTreeClassifier: ')
y_pred = predict_test(X_scaled_test, classifier)
accuracy = accuracy_print_conf(y_test, y_pred)
springFS_pred_test = y_pred.copy()

# =============================================================================
# SVR Preparing
# =============================================================================
# Switch of 1day, 2days or 3days
#X = day0.copy()
#X = np.c_[day0, day_1]
X = np.c_[day0, day_1, day_2]

# Transfer to dataframe and seperate to train, valid and test set
X = pd.DataFrame(X, index=X[:, 8])
X.dropna(inplace=True)

# One_Hot Incoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = pd.DataFrame(X, index=X[:, 19])

test = X.loc['2013-01-01':'2013-12-31'].drop(columns=[19]).values
X = X.loc['1992-01-01':'2013-01-01'].drop(columns=[19]).values#Changed

flowrate.index = flowrate['Datetime']
y_test = flowrate['flow'].loc['2013-01-01':'2013-12-31'].values
y = flowrate['flow'].loc['1992-01-01':'2013-01-01'].values

# Feature scaling for training set
X_month = X[:, :12]
X_SF = X[:, 19]
X = np.c_[X[:, 12:19], X[:, 20:]]

sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
X_scaled = np.c_[X_month, X_scaled, X_SF]
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))
y_scaled = y_scaled.reshape(len(y_scaled))


# Feature scaling for test set
X_month_test = test[:, :12]
springFS_real_test = test[:, 19]
test = np.c_[test[:, 12:19], test[:, 20:]]

X_scaled_test = sc_X.transform(test)
X_scaled_test = np.c_[X_month_test, X_scaled_test, springFS_pred_test]

# =============================================================================
# SVR 
# =============================================================================
from sklearn.svm import SVR
regressor = SVR()

parameters = {'kernel':('linear', 'poly', 'rbf'), 
              'C':[1,2,3,4,5,6,7,8,9,10],
              'degree':[2,3,4,5]
              }
clf = GridSearchCV(regressor, parameters,n_jobs=-1, cv=5)
clf.fit(X_scaled, y_scaled)
print('Best score:', clf.best_score_)
print('Best parameters:', clf.best_params_)

regressor = SVR(kernel=clf.best_params_.get('kernel'), C=clf.best_params_.get('C'))
regressor.fit(X_scaled, y_scaled)

# Prediction
y_pred_scaled = regressor.predict(X_scaled_test)
y_pred = sc_y.inverse_transform(y_pred_scaled)

# Comparing y_pred and y_test
rmse = rootMSE(y_test, y_pred)
'''
# Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

parameters = {'n_estimators':[45,50,55,60,70,80,90,100]}
clf = GridSearchCV(regressor, parameters,n_jobs=-1, cv=5)
clf.fit(X_scaled, y_scaled)
print('Best score:', clf.best_score_)
print('Best parameters:', clf.best_params_)
regressor = RandomForestRegressor(n_estimators=clf.best_params_.get('n_estimators'),
                                  n_jobs=-1, random_state=1029)
regressor.fit(X_scaled, y_scaled)

# Prediction
y_pred_scaled = regressor.predict(X_scaled_test)
y_pred = sc_y.inverse_transform(y_pred_scaled)

# Comparing y_pred and y_test
rmse = rootMSE(y_test, y_pred)
'''
# =============================================================================
# Save/Load results
# =============================================================================
from joblib import dump
#dump(regressor, 'FRO_KC1_SVM_defaultHyperpara.joblib')
dump(classifier, 'FRO_KC1_DecisTreeClas.joblib')

#from joblib import load
#regressor = load('FRO_KC1_SVM_defaultHyperpara.joblib') 














