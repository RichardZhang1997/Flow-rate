# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:56:19 2021

@author: Administrator
"""


import os
os.chdir("D:\\Study\\Marko Mine\\Flowrate")

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

station = 'FRO_KC1'
flowrate = pd.read_csv(station+'_.csv', usecols=[2, 3])
threshold = 1.2

# Converting date string to datetime
flowrate['Datetime'] = pd.to_datetime(flowrate['sample_date'], format='%Y/%m/%d')
flowrate = flowrate.drop('sample_date', 1)
flowrate.columns = ['flow', 'sample_date']

fontdict_exmt = {'size':12, 'color':'r', 'family':'Times New Roman'}
fontdict_ctrl = {'size':12, 'color':'g', 'family':'Times New Roman'}
titlefontdic = {'size':16, 'color':'k', 'family':'Times New Roman'}
text_font = {'size':'22', 'color':'black', 'weight':'bold', 'family':'Times New Roman'}
font1={'family': 'Times New Roman', 'weight': 'light', 'size': 12}
font2={'family': 'Times New Roman', 'weight': 'light', 'size': 16}
# =============================================================================
# Flow rate analysis (to determine the threshold)
# =============================================================================
flowrate['month'] = flowrate['sample_date'].map(lambda x: x.month)
flow_mean = flowrate.groupby('month')['flow'].mean()
flow_std = flowrate.groupby('month')['flow'].std()
Month = flow_mean.index

fig, ax = plt.subplots(figsize=(10,5), dpi=150)
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.title(station,fontdict=font2, pad=14)#Title
ax.errorbar(x=flow_mean.index,y=flow_mean,yerr=flow_std,alpha=.7,color='g',label='Flowrate',fmt='o:',mfc='wheat',mec='salmon',capsize=5)
plt.plot([1,12],[threshold,threshold],'--',label='Threshold = '+str(threshold),c='black',lw=1.0)
ax.set_ylabel('Monthly Averaged Flow Rate ($\mathregular{m^{3}}$/s)',fontsize='12')
ax.set_xlabel('Month',fontsize='12')
ax.set_xlim(0,13)
ax.set_xticks(np.arange(0, 13, step=1))
ax.text(0.91, 0.92, '(a)', transform = ax.transAxes, fontdict=text_font, zorder=4)
plt.legend(loc='upper left')
plt.show()
