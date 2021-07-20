# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:01:16 2021

@author: JavierMencíaLedo
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
import pmdarima as pm


df = pd.read_csv('./california.csv', sep=',', parse_dates=['date'], index_col='date')

#Remove empty columns
empty_cols = [col for col in df.columns if df[col].isnull().all()]
# Drop these columns from the dataframe
df.drop(empty_cols,
        axis=1,
        inplace=True)

headers = list(df.columns)
headers[-1]=headers[-1].strip("\n")
boolean, numeric = [],[]
mostunique = [headers[1],0]
for h in headers:
    temp = df[h].dropna()
    temp = [i for i in temp if i is not None]
    
    if (len(set(temp))==2 and len(temp)>2):
        boolean.append(h)
    elif (df[h].dtype == np.int64 or df[h].dtype==np.float64) :
        numeric.append(h)
    else:
        if eval(str(len(set(df[h].tolist())))+"/"+str(len(df[h].tolist())))>mostunique[1]:
            mostunique = [h, eval(str(len(set(df[h].tolist())))+"/"+str(len(df[h].tolist())))]
df = df.loc[:,~df.columns.duplicated()]
dup = df[df.duplicated(keep="first")]
#Processing duplicates and n
dfcopy = df.copy(deep = True)
column_means = df.interpolate()
for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
    if i in boolean:
        df[i].fillna(df[i].mode(),inplace=True)
    else:
        df[i].fillna(df[i].interpolate())#method='spline', order = 3, limit_direction='forward'),inplace=True)
df = df.drop_duplicates()
dfcopy = dfcopy.drop_duplicates()

df = df.dropna()
xfield = mostunique[0]
print(df.to_string())
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(12,5))
df.hist(ax=ax1)
#df.plot(kind='kde', ax=ax2)
plt.show()

for n in numeric[1:]:
    print(n)
    print(df[n].tolist())
    """
    decomposition = seasonal_decompose(df[n], freq=12, model='additive')
    plt.rcParams['figure.figsize'] = 12, 5
    decomposition.plot()
    plt.show()
    
    
    #Determing rolling statistics
    rolmean = pd.Series(df[n]).rolling(window=12).mean()
    rolstd = pd.Series(df[n]).rolling(window=12).std()
    
    #Plot rolling statistics:
    orig = plt.plot(df[n], color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)"""
    
    df_diff = df.diff().diff(12)
    df_diff.dropna(inplace=True)
    dftest = adfuller(df_diff[n])

    
    """dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        
    
    plt.plot(df_diff[n])
    plt.title(n)
    plt.savefig('diffplot')
    plt.show()"""
    
    """fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14,6), sharex=False, sharey=False)
    ax1 = plot_acf(df_diff[n], lags=50, ax=ax1)
    ax2 = plot_pacf(df_diff[n], lags=50, ax=ax2)
    plt.savefig('acfpacf2')
    plt.show()"""
    
    """model = pm.auto_arima(df[n], d=1, D=1,
                      seasonal=True, m=12, trend='c', 
                      start_p=0, start_q=0, max_order=6, test='adf', stepwise=True, trace=True)
    """
    #divide into train and validation set
    train = df[:int(0.85*(len(df)))]
    test = df[int(0.85*(len(df))):]
    
    #plotting the data
    train[n].plot()
    test[n].plot()
    plt.show()
    
    """model = SARIMAX(train[n],order=(1,1,0),seasonal_order=(0,1,1,12))
    results = model.fit()

    results.plot_diagnostics(figsize=(16, 8))
    plt.savefig('modeldiagnostics')
    plt.show()
    
        
    forecast_object = results.get_forecast(steps=len(test))
    
    mean = forecast_object.predicted_mean
    
    conf_int = forecast_object.conf_int()
    
    dates = mean.index
    plt.figure(figsize=(16,8))

    # Plot past CO2 levels
    plt.plot(df.index, df, label='real')
    
    # Plot the prediction means as line
    plt.plot(dates, mean, label='predicted')
    
    # Shade between the confidence intervals
    plt.fill_between(dates, conf_int.iloc[:,0], conf_int.iloc[:,1],
    alpha=0.2)
    
    # Plot legend and show figure
    plt.legend()
    plt.savefig('predtest')
    plt.show()"""