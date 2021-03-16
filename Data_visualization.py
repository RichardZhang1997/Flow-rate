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

###############################################################################
# Loading datasets
#flowrate = pd.read_csv('FRO_KC1_.csv', usecols=[2, 10])
#flowrate = pd.read_csv('FRO_HC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('GHO_CC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('GHO_PC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('EVO_HC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('GHO_SC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('LCO_WLC_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('LCO_LC3_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('EVO_BC1_.csv', usecols=[2, 3])
#flowrate = pd.read_csv('EVO_EC1_.csv', usecols=[2, 3])
flowrate = pd.read_csv('EVO_SM1_.csv', usecols=[2, 3])
###############################################################################
weather = pd.read_csv('en_climate_daily_BC_1157630_1990-2013_P1D.csv', 
                      usecols=[4, 5, 6, 7, 13, 19, 21, 23, 25]) 

# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)

weather['Datetime'] = pd.to_datetime(weather['Date/Time'], format='%Y/%m/%d')
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

#Flowrate
flowrate.dropna(inplace=True)
flowrate['Year'] = flowrate['Datetime'].dt.year.astype('int')
flowrate['Month'] = flowrate['Datetime'].dt.month.astype('int')
flowrate['Day'] = flowrate['Datetime'].dt.day.astype('int')

flowrate.groupby('Year')['flow'].mean().plot.bar(title='Average Flow Rate by Year for Station FRO KC1')
flowrate.groupby('Year')['flow,'].mean().drop(index=[1996,1997,1998,2000,2001,2002,2003,2004]).plot.bar(title='Average Flow Rate by Year for Station FRO HC1')
flowrate.groupby('Year')['flow,'].mean().plot.bar(title='Average Flow Rate by Year for Station GHO CC1')
flowrate.groupby('Year')['flow,'].mean().plot.bar(title='Average Flow Rate by Year for Station GHO PC1')
flowrate.groupby('Year')['flow,'].mean().drop(index=[1992,2000]).plot.bar(title='Average Flow Rate by Year for Station EVO HC1')
flowrate.groupby('Year')['flow,'].mean().drop(index=[1995,2001]).plot.bar(title='Average Flow Rate by Year for Station GHO SC1')
flowrate.groupby('Year')['flow'].mean().drop(index=[2001,2002,2003,2005,2006,2007,2008,2009]).plot.bar(title='Average Flow Rate by Year for Station LCO WLC')
flowrate.groupby('Year')['flow,'].mean().drop(index=[2007,2009]).plot.bar(title='Average Flow Rate by Year for Station LCO LC3')
flowrate.groupby('Year')['report_result_value'].mean().drop(index=[1992,2004]).plot.bar(title='Average Flow Rate by Year for Station EVO BC1')
flowrate.groupby('Year')['report_result_value'].mean().drop(index=[1996,2001,2004]).plot.bar(title='Average Flow Rate by Year for Station EVO EC1')
flowrate.groupby('Year')['report_result_value'].mean().drop(index=[1985,1992,1997,1998,1999,2000,2001,2002,2003]).plot.bar(title='Average Flow Rate by Year for Station EVO SM1')

flowrate.groupby('Month')['report_result_value'].mean().plot.bar(
    title='Average Flow Rate by Month for EVO EC1 Station')

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

temp = weather.groupby('Month')['Total Snow (cm)'].mean()#.plot.pie(title='Average Total Rain (mm) by Month')
plt.pie(x=temp, explode=explode, labels=temp.index, colors=colors,
        autopct='%.1f%%', pctdistance=0.8, labeldistance=1.1, startangle=120,
        radius=1.2, counterclock=False,
        wedgeprops={'linewidth':1.5,'edgecolor':'black'},
        textprops={'fontsize':8, 'color':'black'})#保留一位小数
plt.title('Average Total Snow (cm) by Month')
plt.show()

temp = weather.groupby('Month')['Total Precip (mm)'].mean()#.plot.pie(title='Average Total Rain (mm) by Month')
plt.pie(x=temp, explode=explode, labels=temp.index, colors=colors,
        autopct='%.1f%%', pctdistance=0.8, labeldistance=1.1, startangle=120,
        radius=1.2, counterclock=False,
        wedgeprops={'linewidth':1.5,'edgecolor':'black'},
        textprops={'fontsize':8, 'color':'black'})#保留一位小数
plt.title('Average Total Precipitation (mm) by Month')
plt.show()

temp = weather.groupby('Month')['Snow on Grnd (cm)'].mean()#.plot.pie(title='Average Total Rain (mm) by Month')
plt.pie(x=temp, explode=explode, labels=temp.index, colors=colors,
        autopct='%.1f%%', pctdistance=0.8, labeldistance=1.1, startangle=120,
        radius=1.2, counterclock=False,
        wedgeprops={'linewidth':1.5,'edgecolor':'black'},
        textprops={'fontsize':8, 'color':'black'})#保留一位小数
plt.title('Average Snow Depth (cm) by Month')
plt.show()

plt.hist(x=flowrate['flow'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('Flow Rate Distribution', pad=10)#pad是标题离圆心的距离
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

#flow rate histogram to help to decide what should the threshold of SF be
plt.hist(x=flowrate['report_result_value'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('EVO EC1 Station Flow Rate Distribution', pad=10)#pad是标题离圆心的距离
plt.show()

#Weather histogram 
#Mean Temp (°C)
plt.hist(x=weather['Mean Temp (°C)'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('Mean Temperature Distribution of Sparwood Station', pad=10)#pad是标题离圆心的距离
plt.show()

#Total Rain (mm)
plt.hist(x=weather['Total Rain (mm)'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('Total Rain Distribution of Sparwood Station', pad=10)#pad是标题离圆心的距离
plt.show()

#Total Snow (cm)
plt.hist(x=weather['Total Snow (cm)'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('Total Snow Distribution of Sparwood Station', pad=10)#pad是标题离圆心的距离
plt.show()

#Total Precip (mm)
plt.hist(x=weather['Total Precip (mm)'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('Total Precipitation Distribution of Sparwood Station', pad=10)#pad是标题离圆心的距离
plt.show()

#Snow on Grnd (cm)
plt.hist(x=weather['Snow on Grnd (cm)'], bins=48, color='r', edgecolor='black', density=True)#density means y_axis is the frequency instead of values
#总面积一定
plt.title('Snow Depth Distribution of Sparwood Station', pad=10)#pad是标题离圆心的距离
plt.show()

print(weather.describe())
