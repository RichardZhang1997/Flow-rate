# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:46:39 2021

@author: Administrator
"""

import os
os.chdir("D:\\Study\\Marko Mine\\Flowrate\\Environment Canada\\Weather")

#Import libraries
import pandas as pd
import datetime

# =============================================================================
# Data integration
# =============================================================================
station = 'Sugarloaf LO'
start_year = 1970#include
end_year = 1996#include
# Reading the first file

year_weather = pd.read_csv(station+'\\en_climate_daily_AB_3056250_'+str(start_year)#the province/station name also to be changed
                           +'_P1D.csv',usecols=[4,5,6,7,13,19,21,23,25])
# Appending the following years to the first file
for i in range(start_year+1, end_year+1):
    year_weather = year_weather.append(pd.read_csv(station+'\\en_climate_daily_AB_3056250_'#the province/station name also to be changed
                                                   +str(i)+'_P1D.csv',
                                                   usecols=[4,5,6,7,13,19,21,23,25]))

# Saving the results
pd.DataFrame(year_weather).to_csv(station+'\\en_climate_daily_AB_3056250_'#the province/station name also to be changed
                                  +str(start_year)+'-'+str(end_year)+'_P1D.csv')

# =============================================================================
# Data set construction
# =============================================================================
weather_sparwood = pd.read_csv('Sparwood'+'\\en_climate_daily_BC_1157630_1980-2020_P1D.csv', usecols=[1,5,6,7,8])
weather_sparwood['Datetime'] = pd.to_datetime(weather_sparwood['Date/Time'], format='%Y/%m/%d')
weather_sparwood = weather_sparwood.drop('Date/Time', 1)
weather_sparwood.columns = ['sp_temp','sp_rain','sp_snow','sp_precip','Datetime']
weather_sparwood.index = weather_sparwood['Datetime']

weather_coleman = pd.read_csv('Coleman'+'\\en_climate_daily_AB_3051720_1924-1992_P1D.csv', usecols=[1,5,6,7,8])
weather_coleman['Datetime'] = pd.to_datetime(weather_coleman['Date/Time'], format='%Y/%m/%d')
weather_coleman = weather_coleman.drop('Date/Time', 1)
weather_coleman.columns = ['co_temp','co_rain','co_snow','co_precip','Datetime']
weather_coleman.index = weather_coleman['Datetime']

weather_sugarloaf = pd.read_csv('Sugarloaf LO'+'\\en_climate_daily_AB_3056250_1970-1996_P1D.csv', usecols=[1,5,6,7,8])
weather_sugarloaf['Datetime'] = pd.to_datetime(weather_sugarloaf['Date/Time'], format='%Y/%m/%d')
weather_sugarloaf = weather_sugarloaf.drop('Date/Time', 1)
weather_sugarloaf.columns = ['su_temp','su_rain','su_snow','su_precip','Datetime']
weather_sugarloaf.index = weather_sugarloaf['Datetime']

date = weather_coleman['Datetime']
date = date.append(weather_sparwood['Datetime'].loc[date.index[-1]+datetime.timedelta(days=1):])
date.index = range(0,len(date))

weather_sparwood.index = range(0, len(weather_sparwood))
weather_coleman.index = range(0, len(weather_coleman))
weather_sugarloaf.index = range(0, len(weather_sugarloaf))

weather = pd.merge(pd.DataFrame(date), weather_sparwood, on=('Datetime'), how='left')
weather = pd.merge(weather, weather_coleman, on=('Datetime'), how='left')
weather = pd.merge(weather, weather_sugarloaf, on=('Datetime'), how='left')

weather.to_csv('Integrated Weather.csv')



