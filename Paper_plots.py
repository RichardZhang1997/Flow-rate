# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:42:57 2021

@author: Administrator
"""

import os
os.chdir("D:\\Study\\Marko Mine\\Flowrate")

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.metrics import r2_score

station = 'FRO_HC1'
regression = pd.read_csv('./Visualization/Data/'+station+'_data.csv')
#regression = pd.read_csv('./Visualization/Data/'+station+'_data_new.csv')#only FRO HC1 is better for new one
regression_norm = pd.read_csv('./Visualization/Data/'+station+'_data_norm.csv')
#regression_norm = pd.read_csv('./Visualization/Data/'+station+'_data_norm_new.csv')
threshold = 1.8
#regression Index(['Date', 'Measured', 'Output_withSF', 'Output_withoutSF'], dtype='object')

fontdict_exmt = {'size':12, 'color':'r', 'family':'Times New Roman'}
fontdict_ctrl = {'size':12, 'color':'g', 'family':'Times New Roman'}
titlefontdic = {'size':16, 'color':'k', 'family':'Times New Roman'}
text_font = {'size':'22', 'color':'black', 'weight':'bold', 'family':'Times New Roman'}
font1={'family': 'Times New Roman', 'weight': 'light', 'size': 12}
font2={'family': 'Times New Roman', 'weight': 'light', 'size': 16}

# =============================================================================
# Correlation plots
# =============================================================================
# calculating parameters
#N = len(regression)
x = regression_norm['Measured']
y_exmt = regression_norm['Output_withSF']
y_ctrl = regression_norm['Output_withoutSF']#(x1, y1) is the scatter plot
C_exmt = round(r2_score(x,y_exmt), 4)
C_ctrl = round(r2_score(x,y_ctrl), 4)
#rmse = round(np.sqrt(mean_squared_error(x,y)), 3)

# calculating the plotting data
x2 = np.linspace(-10, 10)
y2 = x2#(x2, y2) is the the diagonal plot
def f_line(x, A, B):#return a value on curve with slope A and bias B
    return A*x+B
slope_1_exmt, bias_1_exmt = optimize.curve_fit(f_line, x, y_exmt)[0]# return the fitting slope and bias
slope_1_ctrl, bias_1_ctrl = optimize.curve_fit(f_line, x, y_ctrl)[0]# return the fitting slope and bias
y3_exmt = slope_1_exmt*x + bias_1_exmt# (x, y3) is the line fitting plot
y3_ctrl = slope_1_ctrl*x + bias_1_ctrl# (x, y3) is the line fitting plot

# plotting 
#fontdict = {'size':16, 'color':'k', 'family':'Times New Roman'}
fig, ax = plt.subplots(figsize=(6,6), dpi=200)
#plt.scatter(x, y_exmt, edgecolors=None, c='r', s=30, marker='s')# edgecolors=None, c='r', s=30, marker='^'
#plt.scatter(x, y_ctrl, edgecolors=None, c='g', s=30, marker='^')# edgecolors=None, c='g', s=30, marker='s'
ax.plot(x, y_exmt, '^', markerfacecolor='none', ms=6, markeredgecolor='red', label='With SF')
ax.plot(x, y_ctrl, 's', markerfacecolor='none', ms=6, markeredgecolor='green', label='Without SF')
ax.plot(x2, y2, color='k', linewidth=1.5, linestyle='--')
ax.plot(x, y3_exmt, color='r', linewidth=2, linestyle='-')
ax.plot(x, y3_ctrl, color='g', linewidth=2, linestyle='-')
ax.set_xlabel('Normalized Measured Flow Rate')
ax.set_ylabel('Normalized Model Output Flow Rate')
ax.grid(False)
ax.set_xlim(-2.0, 6.0)
ax.set_ylim(-2.0, 6.0)
ax.set_xticks(np.arange(-2.0, 6.4, step=0.8))
ax.set_yticks(np.arange(-2.0, 6.4, step=0.8))
# setting scale font
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_color('k')
ax.tick_params(left=True, bottom=True, direction='in', labelsize=14)
# adding the titile
ax.set_title(station, titlefontdic, pad=14)
#ax.set_title()
#ax.text(-1.8, 5.7, 'With SF', fontdict=fontdict_exmt)
#ax.text(-0.4, 5.7, 'Without SF', fontdict=fontdict_ctrl)
ax.text(0.5, 5.6, r'$R=$'+str(round(np.sqrt(C_exmt),3)), fontdict=fontdict_exmt)
ax.text(0.5, 5.2, r'$R=$'+str(round(np.sqrt(C_ctrl),3)), fontdict=fontdict_ctrl)

#ax.text(-1.8, 4.9, r'$Slope=$'+str(round(slope_1_exmt,3)), fontdict=fontdict_exmt)
#ax.text(0.0, 4.9, r'$Slope=$'+str(round(slope_1_ctrl,3)), fontdict=fontdict_ctrl)
#ax.text(-1.8, 4.9, r'$Slope=$'+str(0.921), fontdict=fontdict_exmt)#for FRO_KC1 station only
#ax.text(0.0, 4.9, r'$Slope=$'+str(0.841), fontdict=fontdict_ctrl)

#ax.text(0.91, 0.92, '(a)', transform = ax.transAxes, fontdict=text_font, zorder=4)
ax.text(0.91, 0.92, '(b)', transform = ax.transAxes, fontdict=text_font, zorder=4)
#ax.text(0.91, 0.92, '(c)', transform = ax.transAxes, fontdict=text_font, zorder=4)

plt.legend()
plt.show()

# =============================================================================
# Line plots
# =============================================================================
import datetime
from matplotlib.dates import DateFormatter, MonthLocator
#from dateutil.relativedelta import relativedelta
#Generate Data
'''
train_startDate = '1996-01-01'
endDate = '2013-12-31'
pred_long = pd.read_csv('pred_whole_1990-2013_'+station+'_4Input.csv')
pred_long['Date'] = pd.to_datetime(pred_long['Date'], format='%Y/%m/%d')
pred_long.index = pred_long['Date']
pred_long = pred_long.loc[train_startDate:endDate]
'''
x1 = list(regression['Date'])
x1 = [datetime.datetime.strptime(d, '%Y/%m/%d').date() for d in x1]
x2 = list(regression['Date'])
x2 = [datetime.datetime.strptime(d, '%Y/%m/%d').date() for d in x2]
y1 = list(regression['Measured'])
y2 = list(regression['Output_withSF'])
text_font = {'size':'22', 'color':'black', 'weight':'bold', 'family':'Times New Roman'}

fig, ax = plt.subplots(figsize=(15,5), dpi=200)
plt.rcParams['axes.unicode_minus'] = False#使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman']
#plt.rcParams['font.sans-serif']=['SimHei']
#wight为字体的粗细，可选 ‘normal\bold\light’等
#size为字体大小
plt.title(station,fontdict=font2, pad=14)#Title
plt.scatter(x1, y1, edgecolors=None, c='r', s=15, marker='s', label='Measured')
plt.plot(x2, y2,'b-', lw=1.0, label="Model Output")#Line
plt.axvline(x=datetime.date(2013,1,1), ls='--', c='black', lw=1.0)
plt.ylabel('Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font1)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font1)
plt.legend(loc="upper left",scatterpoints=1,prop=font1,shadow=True,frameon=False)#添加图例,
ax.set_xlim(datetime.date(1992,1,1), datetime.date(2014,1,1))
ax.set_ylim(0.0, 5.0)
# Major ticks every 12 months.
fmt_whole_year = MonthLocator(interval=12)
ax.xaxis.set_major_locator(fmt_whole_year)
date_form = DateFormatter("%Y")#only display year here, capital means 4 digits
ax.xaxis.set_major_formatter(date_form)
#ax.set_xticks(np.arange(datetime.date(1992,1,1), datetime.date(2014,1,2), step=datetime.timedelta(days=365.25)))
#plt.xticks(np.arange(datetime.date(1992,1,1), datetime.date(2014,1,2), step=datetime.timedelta(days=365.25)), rotation=45)
#ax.set_yticks(np.arange(datetime.date(1992,1,1), datetime.date(2014,1,1), step=365))
plt.text(0.40, 0.94, 'Train', fontdict=font2, transform = ax.transAxes)
plt.text(0.96, 0.94, 'Test', fontdict=font2, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(a)', fontdict=text_font, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(b)', fontdict=text_font, transform = ax.transAxes)
plt.text(0.92, 0.92, '(c)', fontdict=text_font, transform = ax.transAxes)
plt.show()



# Visualize the measured flow rate and threshold
fig, ax = plt.subplots(figsize=(15,5), dpi=200)
plt.rcParams['axes.unicode_minus'] = False#使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.title(station,fontdict=font2, pad=14)#Title
#plt.scatter(x1, y1, edgecolors=None, c='b', s=15, marker='o', label='Measured')
plt.plot(x1, y1,'b-', lw=1.0, marker='o', label="Model Output")#Line
plt.plot([x1[0], x1[-1]], [threshold,threshold],'--', c='black', lw=1.0, label="SF Threshold")#Dash line
plt.ylabel('Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font1)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font1)
ax.set_xlim(datetime.date(1992,1,1), datetime.date(2014,1,1))
ax.set_ylim(0.0, 5.0)
# Major ticks every 12 months.
fmt_whole_year = MonthLocator(interval=12)
ax.xaxis.set_major_locator(fmt_whole_year)
date_form = DateFormatter("%Y")#only display year here, capital means 4 digits
ax.xaxis.set_major_formatter(date_form)
plt.legend(loc="upper left",scatterpoints=1,prop=font1,shadow=True,frameon=False)#添加图例,
#plt.text(0.91, 0.92, '(a)', fontdict=text_font, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(b)', fontdict=text_font, transform = ax.transAxes)
plt.text(0.91, 0.92, '(c)', fontdict=text_font, transform = ax.transAxes)
plt.show()



# Sensitivity test plots
sensitivity = pd.read_csv('./Visualization/Data/'+station+'_sensitivity.csv')
'''
Index(['Test', 'Real', 'Baseline', 'temp+20', 'temp-20', 'precip+20',
       'precip-20', 'SF=1', 'SF=0'],
      dtype='object')
'''
sensitivity['Date'] = pd.to_datetime(sensitivity['Test'], format='%Y/%m/%d')
sensitivity.drop('Test',1, inplace=True)
x = list(sensitivity['Date'])
y_base = list(sensitivity['Baseline'])
y_T_plus = list(sensitivity['temp+20'])
y_T_minus = list(sensitivity['temp-20'])
y_P_plus = list(sensitivity['precip+20'])
y_P_minus = list(sensitivity['precip-20'])

fig, ax = plt.subplots(figsize=(7,5), dpi=200)
plt.rcParams['axes.unicode_minus'] = False#使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.title(station,fontdict=font2, pad=14)#Title
#plt.scatter(x1, y1, edgecolors=None, c='b', s=15, marker='o', label='Measured')
plt.plot(x, y_T_plus,'--', color='red', lw=1.0, label="Temperatrue+20%")#Line
plt.plot(x, y_base,'-', color='black', lw=1.0, label="Baseline")#Line
plt.plot(x, y_T_minus,'-.', color='blue', lw=1.0, label="Temperatrue-20%")#Line
plt.ylabel('Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font1)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font1)
plt.legend(loc="upper left",scatterpoints=1,prop=font1,shadow=True,frameon=False)#添加图例,
#ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.2, 4.0)
# Major ticks every 1 months.
fmt_whole_month = MonthLocator(interval=1)
ax.xaxis.set_major_locator(fmt_whole_month)
date_form = DateFormatter("%Y-%m")#only display year here, capital means 4 digits
ax.xaxis.set_major_formatter(date_form)
#plt.text(0.91, 0.92, '(a)', fontdict=text_font, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(b)', fontdict=text_font, transform = ax.transAxes)
plt.text(0.91, 0.92, '(c)', fontdict=text_font, transform = ax.transAxes)
plt.show()

fig, ax = plt.subplots(figsize=(7,5), dpi=200)
plt.rcParams['axes.unicode_minus'] = False#使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.title(station,fontdict=font2, pad=14)#Title
#plt.scatter(x1, y1, edgecolors=None, c='b', s=15, marker='o', label='Measured')
plt.plot(x, y_P_plus,'--', color='maroon', lw=1.0, label="Precipitation+20%")#Line
plt.plot(x, y_base,'-', color='black', lw=1.0, label="Baseline")#Line
plt.plot(x, y_P_minus,'-.', color='green', lw=1.0, label="Precipitation-20%")#Line
plt.ylabel('Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font1)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font1)
plt.legend(loc="upper left",scatterpoints=1,prop=font1,shadow=True,frameon=False)#添加图例,
#ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.2, 4.0)
# Major ticks every 1 months.
fmt_whole_month = MonthLocator(interval=1)
ax.xaxis.set_major_locator(fmt_whole_month)
date_form = DateFormatter("%Y-%m")#only display year here, capital means 4 digits
ax.xaxis.set_major_formatter(date_form)
#`plt.text(0.91, 0.92, '(d)', fontdict=text_font, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(e)', fontdict=text_font, transform = ax.transAxes)
plt.text(0.91, 0.92, '(f)', fontdict=text_font, transform = ax.transAxes)
plt.show()

# =============================================================================
# Bar charts
# =============================================================================
importances = pd.read_csv('./Visualization/Data/FeatureImportance.csv')
importances.index = np.arange(1, len(importances)+1)
stations = np.array(importances['Stations'])

fig, ax = plt.subplots(figsize=(7,5), dpi=200)
plt.bar(x = importances.index.values, height=importances['Month'], color='blue', 
        label='Month', tick_label = stations, width = 0.5)
plt.bar(x = importances.index.values, height=importances['Mean Temp'], color='red', 
        label='Mean Temp (℃)', tick_label = stations, bottom=importances['Month'], width = 0.5)
plt.bar(x = importances.index.values, height=importances['Total Precip (mm)'], color='violet', 
        label='Total Precip (mm)', tick_label = stations, bottom=importances['Mean Temp']+importances['Month'], width = 0.5)
plt.bar(x = importances.index.values, height=importances['Total Rain (mm)'], color='green', 
        label='Total Rain (mm)', tick_label = stations, bottom=importances['Total Precip (mm)']+importances['Mean Temp']+importances['Month'], width = 0.5)
plt.ylabel('Importance Contribution')
#plt.title('Importance Plot')
plt.legend(bbox_to_anchor=(1.01, 0.5))
plt.show()
