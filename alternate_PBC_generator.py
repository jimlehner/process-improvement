# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:24:31 2023

@author: james
"""

import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import plotly.io as io
import numpy as np
from matplotlib import pyplot as plt
io.renderers.default='browser' # This has plotly display plot in browser

def PBC(df, parameter, xtick_labels, title='Process Behavior Chart',
        ylabel='', xlabel='', label_font_size=18,
        calculate_limits='On'):
    
    data = df[parameter]
    labels = df[xtick_labels].tolist()
    
    if calculate_limits == 'On' or calculate_limits == 'on':
        # Calculate basic statistics
        mean = data.mean()
        mR = abs(data.diff())
        AmR = mR.mean()
        # Calclulate process limits
        UPL = mean + (2.66*AmR)
        LPL = mean - (2.66*AmR)
        URL = 3.27*AmR
        LPL = max(LPL,0)
        
    else:
        UPL=UPL
        LPL=LPL
        mean=mean
        
    
    # Specify masking parameters 
    upper_lim = np.ma.masked_where(data < UPL, data)
    lower_lim = np.ma.masked_where(data > LPL, data)
    middle = np.ma.masked_where((data < LPL) | (data > UPL), data)
    
    # Plot data
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(labels ,data)
    ax.plot(labels, data, middle, marker='o', ls='none', color='tab:blue')
    ax.plot(labels, lower_lim, marker='o', ls='none', color='tab:red',
            markeredgecolor='black', markersize=9)
    ax.plot(labels, upper_lim, marker='o', ls='none', color='tab:red',
            markeredgecolor='black', markersize=9)
    # Plot limits and centerline
    ax.axhline(UPL,ls='--',c='red')
    ax.axhline(LPL,ls='--',c='red')
    ax.axhline(mean,ls='--',c='black')
    # Axis visibility
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines[['bottom', 'left']].set_alpha(0.4)
    # Set axes titles
    plt.title(title, fontsize=22)
    plt.ylabel(ylabel, fontsize=label_font_size)
    plt.xlabel(xlabel, fontsize=label_font_size)
    # Show plot
    plt.show()

# Original dataframe
df = pd.DataFrame({'Month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                   'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   'Sales': [10.7, 13.0, 11.4, 11.5, 12.5, 14.1, 14.8, 14.1, 
                   12.6, 16.0, 11.7, 10.6]})

df2 = pd.DataFrame({'Month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                   'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   'Sales': [10.7, 13.0, 0, 11.5, 12.5, 14.1, 14.8, 14.1, 
                   12.6, 16.0, 11.7, 10.6]})

df3 = pd.DataFrame({'Month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                   'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   'Sales': [10.7, 13.0, 0, 11.5, 12.5, 14.1, 14.8, 14.1, 
                   12.6, 16.0, 32, 10.6]})
dataset_url = "https://raw.githubusercontent.com/jimlehner/datasets/main/childhood_poverty_rate_w_TT_2010_to_2020.csv"
df4 = pd.read_csv(dataset_url)
print(df4)
df4['Year'] = df4['Year'].astype('string')
print(df4.dtypes)
#df['Month'] = df['Month'].astype('str')
print(df)
print(df.dtypes)

PBC(df,'Sales','Month')
PBC(df2,'Sales','Month')
PBC(df3,'Sales','Month')
PBC(df4,'Poverty Rate (%)','Year')

# Create PBC using plotly
def plotly_PBC(df,parameter,xtick_labels):
    
    # This is the plotly plot
    data = df[parameter]
    labels = df[xtick_labels].tolist()
    # Calculate basic statistics
    mean = round(data.mean(),2)
    mR = abs(data.diff())
    AmR = mR.mean()
    # Calclulate process limits
    UPL = round(mean + (2.66*AmR),2)
    LPL = round(mean - (2.66*AmR),2)
    URL = round(3.27*AmR,2)
    # Ensure LPL cannot be < 0
    LPL = max(LPL,0)
    # Create limit lists for adding to df
    UPL_col = [UPL]*len(data)
    LPL_col = [LPL]*len(data)
    mean_col = [mean]*len(data)
    df['UPL'] = UPL_col
    df['LPL'] = LPL_col
    df['Mean'] = mean_col
    # Rename parameter column to variable
    #df = df.rename(columns={parameter:'Variable'})
    # Print df
    print(df)
    
    # This is the plotly plot
    fig = px.line(df, x='Month', y=[parameter,'UPL','LPL','Mean'], 
                  color_discrete_sequence=['blue','red','red','black'],
                  line_dash='variable',
                  markers=True)
    # Process limits and centerline
# =============================================================================
#     fig.add_hline(UPL, line_dash="dash", line_color="red")
#     fig.add_hline(LPL, line_dash="dash", line_color="red")
#     fig.add_hline(mean, line_dash="dash", line_color="black")
# =============================================================================
    
    #fig.show()

# =============================================================================
# # This is the plotly plot
# data = df['Sales']
# labels = df['Month'].tolist()
# # Calculate basic statistics
# mean = data.mean()
# mR = abs(data.diff())
# AmR = mR.mean()
# # Calclulate process limits
# UPL = mean + (2.66*AmR)
# LPL = mean - (2.66*AmR)
# URL = 3.27*AmR
# LPL = max(LPL,0)
# 
# fig = px.line(df, x='Month', y='Sales', markers=True)
# # Process limits and centerline
# fig.add_hline(UPL, line_dash="dash", line_color="red",
#               annotation_text='UPL',
#               annotation_position='top')
# fig.add_hline(LPL, line_dash="dash", line_color="red")
# fig.add_hline(mean, line_dash="dash", line_color="black")
# fig.update_annotations(font_size=16)
# fig.show()
# =============================================================================
# =============================================================================
# def plotly_scatter(df,parameter,xtick_labels):
# =============================================================================
       
#plotly_PBC(df3,'Sales','Month')

# =============================================================================
# fig = px.scatter(df3,'Month','Sales',
#                  color=(
#                      df3['Sales'] > df3['UPL']) &
#                  (df3['LPL'] < df3['Sales'])
#                  ).astype('int'), colorscale=[[0,'red'],
#                                               1,'green']
# fig.show()
# 
# =============================================================================
