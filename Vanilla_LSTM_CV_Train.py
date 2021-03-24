# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:21:14 2020

@author: Richa
"""

import os
os.chdir("C:\\MyFile\\Study\\Graduate\\Marko Mine\\Flowrate")

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
# Loading datasets
#flowrate = pd.read_csv('FRO_KC1_.csv', usecols=[2, 10])
#flowrate = pd.read_csv('FRO_HC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('GHO_CC1_.csv', usecols=[2, 3])
flowrate = pd.read_csv('GHO_PC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('EVO_HC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('GHO_SC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('LCO_WLC_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('LCO_LC3_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('EVO_BC1_.csv', usecols=[2, 3])
##flowrate = pd.read_csv('EVO_EC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('EVO_SM1_.csv', usecols=[2, 3])
###############################################################################

flowrate.columns = ['sample_date', 'flow']
# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)

#print(flowrate.describe())

# =============================================================================
# Missing weather data filling
# =============================================================================
try:
    weather = pd.read_csv('Weather_filled.csv').drop('Num', 1)
    weather['Datetime'] = pd.to_datetime(weather['Datetime'], format='%Y/%m/%d')
    print('Filled weather data loaded successfully')
except:
    print('Filled weather data not detected, generating...')
    weather = pd.read_csv('en_climate_daily_BC_1157630_1990-2013_P1D.csv', 
                      usecols=[4, 5, 6, 7, 13, 19, 21, 23, 25]) 
    weather['Datetime'] = pd.to_datetime(weather['Date/Time'], format='%Y/%m/%d')
    weather = weather.drop('Date/Time', 1)
    print(weather.describe())
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
    pd.DataFrame(weather).to_csv('Weather_filled.csv')
    print('Filled weather data saved successfully')

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
    print('Predicted results length:', y_pred.shape)
    y_test = np.array(y_test).reshape(-1, 1)
    print('Real results length:', y_test.shape)
    return rmse

# =============================================================================
# LSTM
# =============================================================================
flowrate.index = range(0, len(flowrate))
merge = pd.merge(weather, flowrate, on=('Datetime'), how='left')
merge = np.array(merge)
merge = pd.DataFrame(merge, index=merge[:, 8])

# =============================================================================
# LSTM continue
# =============================================================================
test = merge.loc['2013-01-01':'2013-12-31'].drop(8,1).values#Test must be the LAST year of the dataset
#valid = merge.loc['2012-01-01':'2012-12-31'].drop(8,1).values
train = merge.loc['1990-01-01':'2013-01-01'].drop(8,1).values#Changed

train = np.array(train[:, :])
test = np.array(test[:, :])
#train = np.array(train[1:, :])
#test = np.array(test[1:, :])

# One_Hot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
train = ct.fit_transform(train)
test = ct.transform(test)
train = np.c_[train[:, :11], train[:, 12:]]
test = np.c_[test[:, :11], test[:, 12:]]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

# Constructing a LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.constraints import max_norm
import tensorflow as tf

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)#默认值0.001，先用默认值确定其他超参，learningRate和epoch一起在下面CVTraining 确定
#@tf.function
def create_LSTM(neurons, dropoutRate, constraints):
    # Ignore the WARNING here, numpy version problem
    
    # Initializing the RNN
    regressor = Sequential()
    #regressor.add(Dropout(rate=0.2))
    '''
    # Adding the first layer of LSTM and some Dropout regularization (to prevent overfitting)
    regressor.add(LSTM(units=neurons, return_sequences=True, recurrent_dropout=dropoutRate, 
                       kernel_constraint=max_norm(constraints), recurrent_constraint=max_norm(constraints), 
                       bias_constraint=max_norm(constraints)))
    
    # Adding a second LSTM layer and some Dropout regulariazation
    regressor.add(LSTM(units=neurons, return_sequences=True, recurrent_dropout=dropoutRate, 
                       kernel_constraint=max_norm(constraints), recurrent_constraint=max_norm(constraints), 
                       bias_constraint=max_norm(constraints)))
    '''
    # Adding the last LSTM layer and some Dropout regulariazation
    regressor.add(LSTM(units=neurons, return_sequences=False, recurrent_dropout=dropoutRate, 
                       kernel_constraint=max_norm(constraints), recurrent_constraint=max_norm(constraints), 
                       bias_constraint=max_norm(constraints)))

    # Adding output layer
    regressor.add(Dense(units=1))# Output layer do not need specify the activation function
    
    # Compiling the RNN by usign right optimizer and right loss function
    regressor.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])#adam to be changed
    return regressor
'''
To be continued: Defining training parameters: 
    epochs_max, batch_size, patience, validation_ratio, cv_num
'''
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
for time_step in (5, 10, 50):
    print('Below are results for time_step:', time_step)
    X_train, y_train, y_train_not_scaled = [], [], []
    for i in range(time_step, len(train)):
        if scaled_train[i][18]>=0:
            X_train.append(scaled_train[i- time_step:i, :-1])
            y_train.append(scaled_train[i, -1])
            y_train_not_scaled.append(train[i, -1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Ignore the WARNING here, numpy version problem
    
    X_test, y_test, y_test_not_scaled = [], [], []
    for i in range(time_step, len(test)):
        if scaled_test[i][18]>=0:
            X_test.append(scaled_test[i- time_step:i, :-1])
            y_test.append(scaled_test[i, -1])
            y_test_not_scaled.append(test[i, -1])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Creating the model
    regressor = KerasRegressor(build_fn=create_LSTM, epochs=50, batch_size=8)
    parameters = {'neurons':(5, 10, 50, 100),
                  'dropoutRate':(0, 0.1, 0.2, 0.3),
                  'constraints':(3, 50, 99)}
    
    clf = GridSearchCV(regressor, parameters, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    print('Best score:', clf.best_score_)
    print('Best parameters:', clf.best_params_)
    
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    params = clf.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    # Predicting on test set
    regressor = create_LSTM(neurons=clf.best_params_.get('neurons'),
                            dropoutRate=clf.best_params_.get('dropoutRate'),
                            constraints=clf.best_params_.get('constraints'))
    regressor.fit(X_train, y_train, epochs=50, batch_size=8)
    y_pred_scaled = regressor.predict(X_test)
    sc_flow = MinMaxScaler(feature_range=(0, 1), copy=True)
    sc_flow.fit_transform(np.array(y_train_not_scaled).reshape(-1, 1))
    y_pred = sc_flow.inverse_transform(y_pred_scaled)
    
    #Evaluation
    rootMSE(y_test_not_scaled, y_pred)

# =============================================================================
# New LSTM
# =============================================================================
time_step = 5
print('Below are results for time_step:', time_step)
X_train, y_train, y_train_not_scaled = [], [], []
for i in range(time_step, len(train)):
    if scaled_train[i][18]>=0:
        X_train.append(scaled_train[i- time_step:i, :-1])
        y_train.append(scaled_train[i, -1])
        y_train_not_scaled.append(train[i, -1])
X_train, y_train = np.array(X_train), np.array(y_train)
# Ignore the WARNING here, numpy version problem
    
X_test, y_test, y_test_not_scaled = [], [], []
for i in range(time_step, len(test)):
    if scaled_test[i][18]>=0:
        X_test.append(scaled_test[i- time_step:i, :-1])
        y_test.append(scaled_test[i, -1])
        y_test_not_scaled.append(test[i, -1])
X_test, y_test = np.array(X_test), np.array(y_test)
'''
from sklearn.model_selection import KFold
kfold = KFold(5, shuffle=True, random_state=seed)
for train, valid in kfold.split(X_train):
    print('train: %s, test: %s' %(X_train[train], X_train[valid]))
'''

'''
To be continued: KFold validation for LSTM
'''

'''
try:
    best_neurons = clf.best_params_.get('neurons')
    best_dropoutRate = clf.best_params_.get('dropoutRate')
    best_constraints = clf.best_params_.get('constraints')
except:
    best_neurons = 50
    best_dropoutRate = 0.1
    best_constraints = 99
'''

best_neurons = 5
best_dropoutRate = 0.1
best_constraints = 99

# Creating the model
regressor = create_LSTM(neurons=best_neurons,
                        dropoutRate=best_dropoutRate,
                        constraints=best_constraints)
#r = regressor.fit(X_train, y_train, epochs=50, batch_size=8)
# Using early stopping to train the model
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, 
                                            min_delta=0, restore_best_weights=True)#change patience number
r = regressor.fit(X_train, y_train, epochs=500, batch_size=16, 
              callbacks=[early_stop_callback], validation_split=0.2)#转换成在validation set 上面验证
#Stopped val_loss=0.0036
#接住这个返回值可以画出loss曲线，用尽量大的epoch画图，看是否只是fluctuation
# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

if early_stop_callback.stopped_epoch == 0:
    early_epoch = 500
else:
    early_epoch = early_stop_callback.stopped_epoch
    
print('The training stopped at epoch:', early_epoch)
print('Training the LSTM without monitoring the validation set...')
regressor = create_LSTM(neurons=best_neurons,
                        dropoutRate=best_dropoutRate,
                        constraints=best_constraints)
r = regressor.fit(X_train, y_train, epochs=early_epoch, batch_size=16)
plt.plot(r.history['loss'], label='loss')
plt.legend()
plt.show()

y_pred_scaled = regressor.predict(X_test)
sc_flow = MinMaxScaler(feature_range=(0, 1), copy=True)
sc_flow.fit_transform(np.array(y_train_not_scaled).reshape(-1, 1))
y_pred = sc_flow.inverse_transform(y_pred_scaled)

#Evaluation
rootMSE(y_test_not_scaled, y_pred)

np.savetxt('GHO_PC1_Test_Data_withoutSF.csv',np.c_[y_test_not_scaled,y_pred],fmt='%f',delimiter=',')

# =============================================================================
# Saving the training results
# =============================================================================
regressor.save_weights('./FRO_KC1_')
# Restore the weights
#regressor.load_weights('./FRO_KC1_')#Skip compiling and fitting process

# =============================================================================
# Predicting on 2013 everyday weather data
# =============================================================================
weather_2013 = pd.read_csv('Weather_filled2013.csv').drop('Num', 1).drop('Datetime', 1)
weather_2013 = np.array(weather_2013)

X_test_DT = weather_2013[:,1:]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test_DT = np.c_[weather_2013[1:,:], weather_2013[:-1, 3:]]
X_test = ct.fit_transform(X_test_DT)
X_month = X_test[:, :11]
X_test = np.c_[X_month, weather_2013[1:, 0], weather_2013[1:,2: ]]

test = np.c_[X_test, np.zeros(len(X_test))]
scaled_test = scaler.transform(test)
scaled_test = scaled_test[:, :-1]

time_step = 5
print('Below are results for time_step:', time_step)
X_test = []
for i in range(time_step, len(scaled_test)):
    X_test.append(scaled_test[i- time_step:i, :])

X_test = np.array(X_test)

y_pred_scaled = regressor.predict(X_test)
sc_flow = MinMaxScaler(feature_range=(0, 1), copy=True)
sc_flow.fit_transform(np.array(y_train_not_scaled).reshape(-1, 1))
y_pred = sc_flow.inverse_transform(y_pred_scaled)

#Saving predicted results
np.savetxt('temp_pred_whole_2013_withoutSF.csv',y_pred,fmt='%f',delimiter=',')
