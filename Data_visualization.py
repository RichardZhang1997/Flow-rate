# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:27:57 2020

@author: Richa
"""

import os
os.chdir("C:\\MyFile\\Study\\Graduate\\Marko MIne\\Flowrate")

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
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

plt.hist(x=flowrate['flow'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('Flow rate distribution', pad=10)#pad是标题离圆心的距离
plt.show()

# =============================================================================
# 12 months trend yearly
# =============================================================================

#for i in weather.groupby('Year'):
#    print(i)

data = weather.drop('Datetime', 1).groupby(['Year', 'Month']).mean()

data = np.array(data)
data_1 = []
for i in range(1, 25):#Number of years: total 25 years
    data_1.append(data[(i-1)*12:i*12,:])
data_1 = np.array(data_1)

temp = data_1[:,:,1]
rain = data_1[:,:,2]
snow = data_1[:,:,3]
precip = data_1[:,:,4]
snow_on_ground = data_1[:,:,5]
