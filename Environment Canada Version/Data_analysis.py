# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:20:06 2021

@author: Administrator
"""

import os
os.chdir("D:\\Study\\Marko Mine\\Flowrate")

import pandas as pd

station = '08NG002'
flowrate = pd.read_csv('Environment Canada\\Flowrate\\'+station+'.csv', usecols=[2, 3], skiprows=[0])

flowrate.columns = ['sample_date', 'flow']
# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)
flowrate.dropna(inplace = True)

start_year = min(flowrate.Datetime.dt.year)
end_year = max(flowrate.Datetime.dt.year)

# Initializing
spring_freshet = pd.DataFrame()
current = flowrate.iloc[0, :].copy()
current_year = start_year

for j in range(1, len(flowrate)):
    if flowrate.Datetime.dt.year.iloc[j] > current_year:
        spring_freshet = spring_freshet.append(current)
        current['flow'] = flowrate.iloc[j,0]
        current['Datetime'] = flowrate.iloc[j,1]
        current_year = flowrate.Datetime.dt.year.iloc[j]
        continue
    elif flowrate.iloc[j, 0] > current['flow']:
        current['flow'] = flowrate.iloc[j,0]
        current['Datetime'] = flowrate.iloc[j,1]

pd.DataFrame(spring_freshet).to_csv('Environment Canada\\Flowrate\\SpringFreshet\\SpringFreshetRecordFor_' + station + '.csv')
