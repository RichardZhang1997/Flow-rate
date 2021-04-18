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
flowrate = pd.read_csv('FRO_KC1_.csv', usecols=[2, 10])
#flowrate = pd.read_csv('FRO_HC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('GHO_CC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('GHO_PC1_.csv', usecols=[2, 3])
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
# Generating melting data
# =============================================================================
flowrate_threshold = 2
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
X.dropna(inplace=True)

X_test = X.loc['2013-01-01':'2013-12-31'].values
X = X.loc['1990-01-01':'2013-01-01'].values#Changed
datetime = X[:, 7]
y = X[:, 8]

X = X[:, 1:7]
#X = np.c_[X[:, 1:7], X[:, 9:]]#for more than 2 days
#eliminate 'year' from the inputX = X[:, 1:8] 

# =============================================================================
# Transforming test set
# =============================================================================
datetime_test = X_test[:, 7]
y_test = X_test[:, 8]

X_test = X_test[:, 1:7]
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
# Decision Tree
# =============================================================================
# Fixing a seed
seed = 1029
np.random.seed(seed)
try:
    from joblib import load
    classifier = load('DecisionTreeForLSTM_FRO_KC1.joblib')
    print('Trained decision tree result loaded successfully')
except:
    print('No training result detected, training...')
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()# Use 'entropy' instead of 'gini'
    
    # Grid searching
    from sklearn.model_selection import GridSearchCV
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
    dump(classifier, 'DecisionTreeForLSTM_FRO_KC1.joblib')#To be changed
    print('Decision tree training result saved')

# Prediction
print('For DecisionTreeClassifier: ')
y_pred = predict_test(X_test, classifier)
accuracy = accuracy_print_conf(y_test, y_pred)

#Indicator calculating
from sklearn.metrics import roc_auc_score, classification_report
dt_roc_auc = roc_auc_score(np.int32(y_test), y_pred)
print ("Decision Tree AUC = %2.2f" % dt_roc_auc)
print(classification_report(np.int32(y_test), y_pred))
#springFS_pred_test = y_pred.copy()
'''
# =============================================================================
# Visualization of the tree
# =============================================================================
from sklearn.tree import export_graphviz
from IPython.display import Image
from six import StringIO
import pydotplus
# Need to install GraphViz and pydotplus
feature_names = pd.DataFrame(weather.columns[1:-1])#eliminate 'year' feature name
#feature_names = feature_names.append(pd.DataFrame(weather.columns[3:-1]))#for more than 1 day
feature_names = np.array(feature_names).tolist()
# 文件缓存
dot_data = StringIO()
# 将决策树导入到dot中
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names = feature_names,
                class_names=['NotSF','SF'])
# 将生成的dot文件生成graph
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# 将结果存入到png文件中
graph.write_png('Visualization\Decision_tree_evaluation\Decision_tree_LCO_LC3.png')
# 显示
Image(graph.create_png())
'''

# =============================================================================
# LSTM
# =============================================================================
flowrate.index = range(0, len(flowrate))
merge = pd.merge(weather, flowrate, on=('Datetime'), how='left')
merge = np.array(merge)
merge = np.c_[merge[:, :9], merge[:, 10], merge[:, 9]]#将melt与flowrate列互换
merge = pd.DataFrame(merge, index=merge[:, 8])

'''
# =============================================================================
# EDA
# =============================================================================
feature_names = pd.DataFrame(weather.columns[:-1])
feature_names = np.array(feature_names)
feature_names = np.append(feature_names,'Flowrate\n(m^3/s)')

#Correlation analysis
import seaborn as sns
corr = merge.drop(8,1).drop(9,1).apply(lambda x:x.astype(float)).corr()
sns.heatmap(corr,xticklabels=feature_names,yticklabels=feature_names)

#Features importance analysis of decision tree
feature_names = feature_names[1:-1]
importances = classifier.feature_importances_#get importance
indices = np.argsort(importances)[::-1]#get the order of features
plt.figure(figsize=(12,6))
plt.title("Feature importance by Decision Tree of LCO LC3 Station")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feature_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

#ROC plot
from sklearn.metrics import roc_curve
dt_fpr, dt_tpr, dt_thresholds = roc_curve(np.int32(y_test), classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc, marker='o')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph of LCO LC3 Station')
plt.legend(loc="lower right")
plt.show()
'''
'''
sklearn.metrics.roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)
y_scorendarray of shape (n_samples,)
Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
thresholdsndarray of shape = (n_thresholds,)
Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.
'''

# =============================================================================
# LSTM continue
# =============================================================================
test = merge.loc['2012-12-04':'2013-12-31'].drop(8,1).drop(2,1).values
#valid = merge.loc['2012-01-01':'2012-12-31'].drop(8,1).values
train = merge.loc['1990-01-01':'2012-12-04'].drop(8,1).drop(2,1).values#Changed

# Building X for decision tree
X_DT = np.array(train[:,1:7])#eliminate 'year', 'day' features
X_test_DT = np.array(test[:,1:7])#eliminate 'year', day' features
#X_DT = np.c_[train[1:,1:7], train[:-1, 2:7]]#for more than one days
#X_test_DT = np.c_[test[1:,1:7], test[:-1, 2:7]]#for more than one days

# Predicting spring F.S. by the decision tree classifier
melt_train = classifier.predict(X_DT)
melt_test = classifier.predict(X_test_DT)

train[:, 7] = melt_train
test[:, 7] = melt_test
#train[1:, 8] = melt_train
#test[1:, 8] = melt_test
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
train = np.c_[train[:, :11], train[:, 12:]]#Delete the last column as a dummy variable
test = np.c_[test[:, :11], test[:, 12:]]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)

# Constructing a LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.constraints import max_norm
import tensorflow as tf

#print(tf.__version__)
#tf.keras.backend.clear_session()
#tf.random.set_seed(seed)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)#默认值0.001，先用默认值确定其他超参，learningRate和epoch一起在下面CV_Training确定
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
    validation_ratio, cv_num
'''
#Defining training parameters
gap_days = 0 #No. of days between the last day of input and the predict date

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
for time_step in (5, 10, 50):
    print('Below are results for time_step:', time_step)
    X_train, y_train, y_train_not_scaled = [], [], []
    for i in range(time_step, len(train)):
        if scaled_train[i][18]>=0:
            X_train.append(scaled_train[i- time_step+1-gap_days:i+1-gap_days, :-1])
            y_train.append(scaled_train[i, -1])
            y_train_not_scaled.append(train[i, -1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Ignore the WARNING here, numpy version problem
    
    X_test, y_test, y_test_not_scaled = [], [], []
    for i in range(time_step, len(test)):
        if scaled_test[i][18]>=0:
            X_test.append(scaled_test[i- time_step+1-gap_days:i+1-gap_days, :-1])
            y_test.append(scaled_test[i, -1])
            y_test_not_scaled.append(test[i, -1])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Creating the model
    regressor = KerasRegressor(build_fn=create_LSTM, epochs=50, batch_size=8)#Default CV parameters, not that accurate
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
time_step = 50
print('Below are results for time_step:', time_step)
X_train, y_train, y_train_not_scaled = [], [], []
for i in range(time_step, len(train)):
    if scaled_train[i][18]>=0:
        X_train.append(scaled_train[i- time_step+1-gap_days:i+1-gap_days, :-1])
        y_train.append(scaled_train[i, -1])
        y_train_not_scaled.append(train[i, -1])
X_train, y_train = np.array(X_train), np.array(y_train)
# Ignore the WARNING here, numpy version problem
    
X_test, y_test, y_test_not_scaled = [], [], []
for i in range(time_step, len(test)):
    if scaled_test[i][18]>=0:
        X_test.append(scaled_test[i- time_step+1-gap_days:i+1-gap_days, :-1])
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

best_neurons = 50
best_dropoutRate = 0.1
best_constraints = 99

epochs_max = 50
batch_size = 8
patience = 5

# Creating the model
regressor = create_LSTM(neurons=best_neurons,
                        dropoutRate=best_dropoutRate,
                        constraints=best_constraints)
#r = regressor.fit(X_train, y_train, epochs=50, batch_size=8)
# Using early stopping to train the model
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, 
                                            min_delta=0, restore_best_weights=True)#change patience number
r = regressor.fit(X_train, y_train, epochs=epochs_max, batch_size=batch_size, 
              callbacks=[early_stop_callback], validation_split=0.2)#转换成在validation set 上面验证

# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

if early_stop_callback.stopped_epoch == 0:
    early_epoch = epochs_max
else:
    early_epoch = early_stop_callback.stopped_epoch
    
print('The training stopped at epoch:', early_epoch)
print('Training the LSTM without monitoring the validation set...')
regressor = create_LSTM(neurons=best_neurons,
                        dropoutRate=best_dropoutRate,
                        constraints=best_constraints)
r = regressor.fit(X_train, y_train, epochs=early_epoch, batch_size=batch_size)
plt.plot(r.history['loss'], label='loss')
plt.legend()
plt.show()

y_pred_scaled = regressor.predict(X_test)
sc_flow = MinMaxScaler(feature_range=(0, 1), copy=True)
sc_flow.fit_transform(np.array(y_train_not_scaled).reshape(-1, 1))
y_pred = sc_flow.inverse_transform(y_pred_scaled)

#Evaluation
rootMSE(y_test_not_scaled, y_pred)

np.savetxt('FRO_KC1_Test_Data.csv',np.c_[y_test_not_scaled,y_pred],fmt='%f',delimiter=',')

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

X_test_DT = np.c_[weather_2013[:,1:2], weather_2013[:,3:]]
melt_test = classifier.predict(X_test_DT)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test_DT = np.c_[weather_2013[1:,:], weather_2013[:-1, 3:]]
X_test = ct.fit_transform(X_test_DT)
X_month = X_test[:, :11]
X_test = np.c_[X_month, weather_2013[1:, 0], weather_2013[1:,3: ], melt_test[1:]]

test = np.c_[X_test, np.zeros(len(X_test))]
scaled_test = scaler.transform(test)
scaled_test = scaled_test[:, :-1]

time_step = 50
print('Below are results for time_step:', time_step)
X_test = []
for i in range(time_step, len(scaled_test)):
    X_test.append(scaled_test[i- time_step+1-gap_days:i+1-gap_days, :])

X_test = np.array(X_test)

y_pred_scaled = regressor.predict(X_test)
sc_flow = MinMaxScaler(feature_range=(0, 1), copy=True)
sc_flow.fit_transform(np.array(y_train_not_scaled).reshape(-1, 1))
y_pred = sc_flow.inverse_transform(y_pred_scaled)

#Saving predicted results
np.savetxt('temp_pred_whole_2013.csv',y_pred,fmt='%f',delimiter=',')
