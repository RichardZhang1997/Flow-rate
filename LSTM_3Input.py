# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:18:39 2021

@author: Administrator
"""

import os
os.chdir("D:\\Study\\Marko Mine\\Flowrate")

# Importing the libraries
import numpy as np
import pandas as pd

# =============================================================================
# Loading datasets
# =============================================================================
station = 'EVO_HC1'
flowrate = pd.read_csv(station+'_.csv', usecols=[2, 3])

# =============================================================================
# Choosing parameters
# =============================================================================
avg_days = 6
time_step = 10
gap_days = 0
seed = 1

train_startDate = '1990-01-01'
test_startDate = '2013-01-01'
endDate = '2013-12-31'

# =============================================================================
# Defining functions
# =============================================================================
'''
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
'''
def rootMSE(y_test, y_pred):
    import math
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE = %2.2f' % rmse)
    print('Predicted results length:', y_pred.shape)
    y_test = np.array(y_test).reshape(-1, 1)
    print('Real results length:', y_test.shape)
    return rmse
# =============================================================================
flowrate.columns = ['sample_date', 'flow']
# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)
#print(flowrate.describe())

weather = pd.read_csv('weather_1990-2013_avg_' + str(avg_days) + '.csv')
weather['Datetime'] = pd.to_datetime(weather['Date/Time'], format='%Y/%m/%d')
weather = weather.drop('Date/Time', 1)
#Datetime = weather['Datetime'].copy()

merge = pd.merge(weather, flowrate, on=('Datetime'), how='left')
merge.index = merge['Datetime']

test = merge.loc[test_startDate : endDate].values
train = merge.loc[train_startDate : test_startDate].values

datetime_test = merge.loc[test_startDate : endDate].index[:].strftime('%Y-%m-%d')
datetime_train = merge.loc[train_startDate : test_startDate].index[:].strftime('%Y-%m-%d')

# 3 inputs are year, mean temperature, and total precipitate
test = np.c_[test[:, 0], test[:, 3], test[:, 6], test[:, 9]]
train = np.c_[train[:, 0], train[:, 3], train[:, 6], train[:, 9]]

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

scaled = np.r_[scaled_train, scaled_test]
original = np.r_[train, test]
datetime = np.r_[datetime_train, datetime_test]

print('Below are results for time_step:', time_step)
X_scaled, y_scaled, y_not_scaled= [], [], []
datetime_deNull = []
for i in range(time_step, len(scaled)):
    if scaled[i][-1]>=0:
        sample_input = []
        for j in range(0, time_step):
            sample_input.append(scaled[i-gap_days-(time_step-1-j)*avg_days, :-1])
        X_scaled.append(sample_input)
        y_scaled.append(scaled[i, -1])
        y_not_scaled.append(original[i, -1])
        datetime_deNull.append(datetime[i])
X_scaled, y_scaled, y_not_scaled = np.array(X_scaled), np.array(y_scaled), np.array(y_not_scaled)
datetime_deNull = np.array(datetime_deNull)

test_size = len(pd.DataFrame(test).dropna())#Number of valid test size

X_train = X_scaled[:len(X_scaled)+1-test_size, :, :]
y_train = y_scaled[:len(X_scaled)+1-test_size]
y_train_not_scaled = y_not_scaled[:len(X_scaled)+1-test_size]
train_datetime = datetime_deNull[:len(X_scaled)+1-test_size]

X_test = X_scaled[len(X_scaled)+1-test_size:, :, :]
y_test = y_scaled[len(X_scaled)+1-test_size:]
y_test_not_scaled = y_not_scaled[len(X_scaled)+1-test_size:]
test_datetime = datetime_deNull[len(X_scaled)+1-test_size:]

# Deleting NaNs in samples
k = 0
for i in range(0, len(X_train)):
    if k>=len(X_train):
        break
    for j in X_train[k, :, :]:
        if np.isnan(j[0]) or np.isnan(j[1]) or np.isnan(j[2]):
            #print('k:', k)# for testing print out which sample contains NaN
            X_train = np.r_[X_train[:k, :, :], X_train[k+1:, :, :]]
            y_train = np.r_[y_train[:k], y_train[k+1:]]
            y_train_not_scaled = np.r_[y_train_not_scaled[:k], y_train_not_scaled[k+1:]]
            train_datetime = np.r_[train_datetime[:k], train_datetime[k+1:]]
            k = k - 1
            break
    k = k + 1
# 40/380 are NaNs
k = 0
for i in range(0, len(X_test)):
    if k>=len(X_test):
        break
    for j in X_test[k, :, :]:
        if np.isnan(j[0]) or np.isnan(j[1]) or np.isnan(j[2]):
            #print('k:', k)
            X_test = np.r_[X_test[:k, :, :], X_test[k+1:, :, :]]
            y_test = np.r_[y_test[:k], y_test[k+1:]]
            y_test_not_scaled = np.r_[y_test_not_scaled[:k], y_test_not_scaled[k+1:]]
            test_datetime = np.r_[test_datetime[:k], test_datetime[k+1:]]
            k = k - 1
            break
    k = k + 1
# 0/22 are NaNs

# Constructing a LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.constraints import max_norm
import tensorflow as tf
import matplotlib.pyplot as plt

#print(tf.__version__)
tf.keras.backend.clear_session()
tf.random.set_seed(seed)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)#默认值0.001，先用默认值确定其他超参，learningRate和epoch一起在下面CV_Training确定

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
    regressor.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='relu'))# Output layer do not need specify the activation function
    
    # Compiling the RNN by usign right optimizer and right loss function
    regressor.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])#adam to be changed
    return regressor

#Defining training parameters
best_neurons = 50
best_dropoutRate = 0.1
constraints = 3

#epochs_max = 500
batch_size = 4
#patience = 20

'''
# Creating the model
regressor = create_LSTM(neurons=best_neurons,
                        dropoutRate=best_dropoutRate,
                        constraints=constraints)
#r = regressor.fit(X_train, y_train, epochs=50, batch_size=8)
# Using early stopping to train the model
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, 
                                            min_delta=0, restore_best_weights=True)#change patience number
r = regressor.fit(X_train, y_train, epochs=epochs_max, batch_size=batch_size, 
              callbacks=[early_stop_callback], validation_split=0.2)#转换成在validation set 上面验证
regressor.summary()
# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

if early_stop_callback.stopped_epoch == 0:
    early_epoch = epochs_max
else:
    early_epoch = early_stop_callback.stopped_epoch
'''

early_epoch = 85

print('The training stopped at epoch:', early_epoch)
print('Training the LSTM without monitoring the validation set...')
regressor = create_LSTM(neurons=best_neurons,
                        dropoutRate=best_dropoutRate,
                        constraints=constraints)

r = regressor.fit(X_train, y_train, epochs=early_epoch, batch_size=batch_size, 
                  validation_data=(X_test, y_test), validation_freq=5)
regressor.summary()

plt.plot(range(1,early_epoch+1), r.history['loss'], label='loss')
plt.plot(np.linspace(0,100,21,endpoint=True)[1:int(int(early_epoch/5)+1)], 
         r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

sc_flow = MinMaxScaler(feature_range=(0, 1), copy=True)
sc_flow.fit_transform(np.array(y_train_not_scaled).reshape(-1, 1))

y_pred_scaled = regressor.predict(X_test)
y_pred = sc_flow.inverse_transform(y_pred_scaled)

y_pred_scaled_train = regressor.predict(X_train)
y_pred_train = sc_flow.inverse_transform(y_pred_scaled_train)

#Evaluation
rootMSE(y_test_not_scaled, y_pred)

# =============================================================================
# Plotting the training and test prediction
# =============================================================================
plt.plot(test_datetime, y_test_not_scaled, label='test')
plt.plot(test_datetime, y_pred, label='test pred')#x-label requires turning angle
plt.legend(loc='best')
plt.show()

plt.plot(train_datetime, y_pred_train, label='train pred')
plt.plot(train_datetime, y_train_not_scaled, label='train')
#plt.xticks(train_datetime, train_datetime, rotation = 'vertical')
plt.legend(loc='best')
plt.show()

# =============================================================================
# Saving the training results
# =============================================================================
# Saving prediction on test set
np.savetxt(station+'_Test_Data_ctrl_group.csv',np.c_[test_datetime,y_test_not_scaled,y_pred],fmt='%s',delimiter=',')

# Saving prediction on train set
np.savetxt(station+'_Train_Data_ctrl_group.csv',np.c_[train_datetime,y_train_not_scaled,y_pred_train],fmt='%s',delimiter=',')

# Saving the LSTM weights
regressor.save_weights('./LSTM results/'+station+'_3Input')

# Restore the weights
#regressor.load_weights('./LSTM results/'+station+'_3Input')#Skip compiling and fitting process

# =============================================================================
# Predicting on everyday weather data
# =============================================================================
weather_dense = pd.read_csv('weather_1990-2013_avg_' + str(avg_days) + '.csv')
weather_dense['Datetime'] = pd.to_datetime(weather_dense['Date/Time'], format='%Y/%m/%d')
datetime = weather_dense['Datetime']

weather_dense = weather_dense.drop('Date/Time', 1)
weather_dense = np.array(weather_dense)
X_test = np.c_[weather_dense[1:, 0], weather_dense[1:,3], weather_dense[1:,6]]
#X_test = np.c_[X_test[:, 0:2], X_test[:, 4], X_test[:, 6:]]

test = np.c_[X_test, np.zeros(len(X_test))]
scaled_test = scaler.transform(test)
scaled_test = scaled_test[:, :-1]

print('Below are results for time_step:', time_step)
X_scaled = []
datetime_deNull = []
for i in range(time_step, len(scaled_test)):
    if scaled_test[i][-1]>=0:
        sample_input = []
        for j in range(0, time_step):
            sample_input.append(scaled_test[i-gap_days-(time_step-1-j)*avg_days, : ])
        X_scaled.append(sample_input)
        datetime_deNull.append(datetime[i])
X_scaled = np.array(X_scaled)
test_datetime = np.array(datetime_deNull)
#8754 samples

# Deleting NaNs in samples
k = 0
for i in range(0, len(X_scaled)):
    if k>=len(X_scaled):
        break
    for j in X_scaled[k, :, :]:
        if np.isnan(j[0]) or np.isnan(j[1]) or np.isnan(j[2]):
            #print('k:', k)# for testing print out which sample contains NaN
            X_scaled = np.r_[X_scaled[:k, :, :], X_scaled[k+1:, :, :]]
            test_datetime = np.r_[test_datetime[:k], test_datetime[k+1:]]
            k = k - 1
            break
    k = k + 1
# 7808 available for training

y_pred_scaled = regressor.predict(X_scaled)
sc_flow = MinMaxScaler(feature_range=(0, 1), copy=True)
sc_flow.fit_transform(np.array(y_train_not_scaled).reshape(-1, 1))
y_pred = sc_flow.inverse_transform(y_pred_scaled)

#Saving predicted results
np.savetxt('pred_whole_1990-2013_'+station+'_3Input.csv',np.c_[test_datetime.reshape(-1, 1),y_pred],fmt='%s',delimiter=',')#test_datetime as x
