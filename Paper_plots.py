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

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
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
fig, ax = plt.subplots(figsize=(6,6), dpi=300)
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
#plt.scatter(x, y_exmt, edgecolors=None, c='r', s=30, marker='s')# edgecolors=None, c='r', s=30, marker='^'
#plt.scatter(x, y_ctrl, edgecolors=None, c='g', s=30, marker='^')# edgecolors=None, c='g', s=30, marker='s'
ax.plot(x, y_exmt, '^', markerfacecolor='none', ms=6, markeredgecolor='red', label='With spring freshet')
ax.plot(x, y_ctrl, 's', markerfacecolor='none', ms=6, markeredgecolor='green', label='Without spring freshet')
ax.plot(x2, y2, color='k', linewidth=1.5, linestyle='--')
ax.plot(x, y3_exmt, color='r', linewidth=2, linestyle='-')
ax.plot(x, y3_ctrl, color='g', linewidth=2, linestyle='-')
ax.set_xlabel('Normalized measured flow rate',fontdict=font2)
ax.set_ylabel('Normalized model output flow rate',fontdict=font2)
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
#ax.set_title('Station 2', titlefontdic, pad=12)
#ax.set_title()
#ax.text(-1.8, 5.7, 'With SF', fontdict=fontdict_exmt)
#ax.text(-0.4, 5.7, 'Without SF', fontdict=fontdict_ctrl)
ax.text(1.8, 5.5, r'$R=$'+str(round(np.sqrt(C_exmt),3)), fontdict=font1)
ax.text(1.8, 5.1, r'$R=$'+str(round(np.sqrt(C_ctrl),3)), fontdict=font1)

#ax.text(-1.8, 4.9, r'$Slope=$'+str(round(slope_1_exmt,3)), fontdict=fontdict_exmt)
#ax.text(0.0, 4.9, r'$Slope=$'+str(round(slope_1_ctrl,3)), fontdict=fontdict_ctrl)
#ax.text(-1.8, 4.9, r'$Slope=$'+str(0.921), fontdict=fontdict_exmt)#for FRO_KC1 station only
#ax.text(0.0, 4.9, r'$Slope=$'+str(0.841), fontdict=fontdict_ctrl)

#ax.text(0.91, 0.92, '(a)', transform = ax.transAxes, fontdict=text_font, zorder=4)
ax.text(0.76, 0.94, 'Station 3', transform = ax.transAxes, fontdict=font2, zorder=4)
#ax.text(0.91, 0.92, '(c)', transform = ax.transAxes, fontdict=text_font, zorder=4)

plt.legend(loc='upper left',frameon=False,prop=font1)
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

fig, ax = plt.subplots(figsize=(15,5), dpi=300)
plt.rcParams['axes.unicode_minus'] = False#使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
#plt.rcParams['font.sans-serif']=['SimHei']
#wight为字体的粗细，可选 ‘normal\bold\light’等
#size为字体大小
#plt.title(station,fontdict=font2, pad=14)#Title
plt.scatter(x1, y1, edgecolors=None, c='r', s=15, marker='s', label='Measured')
plt.plot(x2, y2,'b-', lw=1.0, label="Model Output")#Line
plt.axvline(x=datetime.date(2013,1,1), ls='--', c='black', lw=1.0)
plt.ylabel('Flow rate ($\mathregular{m^{3}}$/s)',fontdict=font2)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font2)
plt.legend(loc="upper left",scatterpoints=1,prop=font1,shadow=True,frameon=False)#添加图例,
ax.set_xlim(datetime.date(1995,1,1), datetime.date(2014,1,1))
ax.set_ylim(0.0, 8.0)
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
plt.text(0.86, 0.94, 'Station 1', fontdict=font2, transform = ax.transAxes)
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
sensitivity = pd.read_csv('./Visualization/Data/'+station+'_sensitivity_1.csv')
'''
Index(['Test', 'Real', 'Baseline', 'temp+20', 'temp-20', 'precip+20',
       'precip-20', 'precip-50', 'precip-50'],
      dtype='object')
'''
sensitivity['Date'] = pd.to_datetime(sensitivity['Test'], format='%Y/%m/%d')
sensitivity.drop('Test',1, inplace=True)
x = list(sensitivity['Date'])
y_base = list(sensitivity['Baseline'])
y_T_plus = list(sensitivity['temp+50'])
y_T_minus = list(sensitivity['temp-50'])
y_P_plus = list(sensitivity['precip+50'])
y_P_minus = list(sensitivity['precip-50'])

fig, ax = plt.subplots(figsize=(7,5), dpi=300)
plt.rcParams['axes.unicode_minus'] = False#使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman']
#plt.title(station,fontdict=font2, pad=14)#Title
#plt.scatter(x1, y1, edgecolors=None, c='b', s=15, marker='o', label='Measured')
plt.plot(x, y_T_plus,'--', color='red', lw=1.0, label="Temperatrue+50%")#Line
plt.plot(x, y_base,'-', color='black', lw=1.0, label="Baseline")#Line
plt.plot(x, y_T_minus,'-.', color='blue', lw=1.0, label="Temperatrue-50%")#Line
plt.ylabel('Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font2)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font2)
plt.legend(loc="upper left",scatterpoints=1,prop=font1,shadow=True,frameon=False)#添加图例,
#ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.2, 4.0)
# Major ticks every 1 months.
fmt_whole_month = MonthLocator(interval=1)
ax.xaxis.set_major_locator(fmt_whole_month)
date_form = DateFormatter("%Y-%m")#only display year here, capital means 4 digits
ax.xaxis.set_major_formatter(date_form)
plt.text(0.82, 0.94, 'Station 3', fontdict=font2, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(b)', fontdict=text_font, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(c)', fontdict=text_font, transform = ax.transAxes)
plt.show()

fig, ax = plt.subplots(figsize=(7,5), dpi=300)
plt.rcParams['axes.unicode_minus'] = False#使用上标小标小一字号
plt.rcParams['font.sans-serif']=['Times New Roman']
#plt.title(station,fontdict=font2, pad=14)#Title
#plt.scatter(x1, y1, edgecolors=None, c='b', s=15, marker='o', label='Measured')
plt.plot(x, y_P_plus,'--', color='maroon', lw=1.0, label="Precipitation+50%")#Line
plt.plot(x, y_base,'-', color='black', lw=1.0, label="Baseline")#Line
plt.plot(x, y_P_minus,'-.', color='green', lw=1.0, label="Precipitation-50%")#Line
plt.ylabel('Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font2)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font2)
plt.legend(loc="upper left",scatterpoints=1,prop=font1,shadow=True,frameon=False)#添加图例,
#ax.set_xlim(x[0], x[-1])
ax.set_ylim(-0.2, 4.0)
# Major ticks every 1 months.
fmt_whole_month = MonthLocator(interval=1)
ax.xaxis.set_major_locator(fmt_whole_month)
date_form = DateFormatter("%Y-%m")#only display year here, capital means 4 digits
ax.xaxis.set_major_formatter(date_form)
plt.text(0.82, 0.94, 'Station 3', fontdict=font2, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(e)', fontdict=text_font, transform = ax.transAxes)
#plt.text(0.91, 0.92, '(f)', fontdict=text_font, transform = ax.transAxes)
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

# =============================================================================
# Flow rate analysis (to determine the threshold)
# =============================================================================
station = 'FRO_KC1'
flowrate = pd.read_csv(station+'_.csv', usecols=[2, 3])
threshold = 1.2

# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)
flowrate.columns = ['flow', 'sample_date']

flowrate['month'] = flowrate['sample_date'].map(lambda x: x.month)

flow_mean = flowrate.groupby('month')['flow'].mean()
flow_std = flowrate.groupby('month')['flow'].std()
flow_min = flowrate.groupby('month')['flow'].min()
flow_max = flowrate.groupby('month')['flow'].max()
y_err = np.c_[np.array(flow_mean-flow_min), np.array(flow_max-flow_mean)].T

fig, ax = plt.subplots(figsize=(10,5), dpi=300)
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['lines.markersize']=8
#plt.title('Station 1',fontdict=font2, pad=12)#Title
ax.errorbar(x=months,y=flow_mean,yerr=y_err,alpha=1,color='black',label='Flowrate',
            fmt='o:',mfc='black',mec='black',capsize=8)
#plt.plot(months, flow_max,'--', color='red', lw=1.0, label="Max Flow Rate")#Line
#plt.plot(months, flow_mean,'-', color='black', lw=1.0, label="Average Flow Rate")#Line
#plt.plot(months, flow_min,'-.', color='blue', lw=1.0, label="Min Flow Rate")#Line
#plt.plot([0,11],[threshold,threshold],'--',label='Threshold = '+str(threshold),c='black',lw=1.0)
plt.plot([0,11],[threshold,threshold],'--',label='Threshold = 0.7',c='black',lw=1.0)
ax.set_ylabel('Monthly flow rate ($\mathregular{m^{3}}$/s)',fontdict=font2)
ax.set_xlabel('Month',fontdict=font2)
ax.set_xlim(-1,12)
ax.set_ylim(-0.2,5)
ax.set_xticks(np.arange(0, 13, step=1))
ax.text(0.82, 0.92, 'Station 3', transform = ax.transAxes, fontdict=font2, zorder=4)
plt.legend(loc='upper left',frameon=False,prop=font1)
plt.show()

# Line-bar plot
plt.rcParams['figure.figsize'] = (10.0,5.0)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(months, flow_std,alpha=.7,color='g', label='Std')
ax1.set_ylabel('Standard Deviation of Flow Rate',fontdict=font2)
plt.legend(loc='upper left',frameon=True,prop=font2)
#plt.title('Station 1',fontdict=font2, pad=12)#Title
ax2 = ax1.twinx()   #组合图必须加这个
ax2.plot(months, flow_mean,'o-', color='black', lw=2.0, label="Average Flow Rate")#Line
ax2.plot([0,11],[threshold,threshold],'--',label='Threshold = '+str(threshold),c='black',lw=1.0)
ax2.set_ylabel('Monthly Averaged Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font2)
ax2.set_xlabel('Month',fontdict=font2)
plt.legend(loc='upper right',frameon=True,prop=font2)
ax2.text(-0.12, 0.9, '(a)', transform = ax.transAxes, fontdict=text_font, zorder=4)
plt.show()

# =============================================================================
# Future Prediction
# =============================================================================
station = 'EVO_HC1'
flowrate = pd.read_csv('./Visualization/Data/Future_pred/'+station+'.csv')
flowrate['Date'] = pd.to_datetime(flowrate['Date'], format='%Y/%m/%d')

fig, ax = plt.subplots(figsize=(10,5), dpi=300)
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.plot(flowrate['Date'], flowrate['Output_withSF'],'-', color='black', lw=1.0)#Line
ax.set_xlim(datetime.date(2014,1,1), datetime.date(2020,1,1))
ax.set_ylim(0.0, 5.0)#5 for station1 and station3, 12 for station2, 
# Major ticks every 12 months.
fmt_whole_year = MonthLocator(interval=12)
ax.xaxis.set_major_locator(fmt_whole_year)
date_form = DateFormatter("%Y")#only display year here, capital means 4 digits
ax.xaxis.set_major_formatter(date_form)
plt.ylabel('Flow Rate ($\mathregular{m^{3}}$/s)',fontdict=font2)#$\mathregular{min^{-1}}$label的格式,^{-1}为上标
plt.xlabel('Time',fontdict=font2)
plt.text(0.86, 0.94, 'Station 3', fontdict=font2, transform = ax.transAxes)
plt.show()

# =============================================================================
# Monthly averaged precipitation and temperature plot
# =============================================================================
weather = pd.read_csv('en_climate_daily_BC_1157630_1990-2013_P1D.csv', 
                      usecols=[4, 5, 6, 7, 13, 19, 21, 23, 25]) 

weather['Datetime'] = pd.to_datetime(weather['Date/Time'], format='%Y/%m/%d')
weather = weather.drop('Date/Time', 1)

mean_temp = weather.groupby('Month')['Mean Temp (°C)'].mean()
std_temp = weather.groupby('Month')['Mean Temp (°C)'].std()
mean_precip = weather.groupby('Month')['Total Precip (mm)'].mean()
std_precip = weather.groupby('Month')['Total Precip (mm)'].std()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_days = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# monthly averaged precipitation and temperature plot
plt.rcParams['figure.figsize'] = (12.0,4.0)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['lines.markersize']=8
fig = plt.figure()
#画柱形图
ax1 = fig.add_subplot(111)
ax1.bar(months, mean_precip*month_days,alpha=.7,color='grey', label='Precipitation',width=0.6)
#ax1.errorbar(x=months,y=mean_precip,yerr=std_precip,alpha=.7,color='g',label='Precipitation',fmt='o:',mfc='wheat',mec='salmon',capsize=5)
ax1.set_ylim(0,100)
ax1.set_ylabel('Average monthly precipitation (mm)',fontdict=font2)
plt.legend(loc='upper left',frameon=False,prop=font2)
#ax1.set_title("数据统计",fontsize='20')
#画折线图 
ax2 = ax1.twinx()   #组合图必须加这个
ax2.plot(months, mean_temp, color='black', linewidth=1, linestyle='-', marker='s', label='Temperature')
#ax2.scatter(months, mean_temp, linewidth=2, color='red', label='Temperature')
ax2.set_ylabel('Average monthly temperature (℃)',fontdict=font2)
ax2.set_ylim(-10,20)
plt.legend(loc='upper right',frameon=False,prop=font2)
plt.show()
