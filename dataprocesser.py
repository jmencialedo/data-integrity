# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:57:27 2021

@author: JavierMencíaLedo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr
from itertools import combinations

def plotter(df,p, corr, maxmin):
    first = p[0]
    second = p[1]
    x,y = df[first].tolist(),df[second].tolist()
    m, b = np.polyfit(x, y, 1)
    plt.plot(x,y,'x')
    corr, p_value = pearsonr(x, y)
    color = corrcolor(corr)
    if abs(corr)>0.5: plt.plot(x, m*np.array(x)+b, color)
    
    plt.xlabel(first)
    plt.ylabel(second)
    if maxmin==1:
        plt.title('Table of two variables with best correlation: ')
        plt.savefig("bestcorrelation.jpg")
    else:
        plt.title('Table of two variables with worst correlation: ')
        plt.savefig("worstcorrelation.jpg")
    plt.show()
    
def corrcolor(corr):
    color = "#ff0000"
    if abs(corr)>0.9: 
        color = "#00ff00"
    elif abs(corr)>0.75: 
        color ="#58ad2a"
    elif abs(corr)>0.6:
        color ="#ffa600"
    return color

def corrmaxmin(df, pairs, maxim):
    if maxim==1:
        result = ["",0]
    else:
        result = ["",1]
    for i in pairs:
        corr, p_value = pearsonr(df[i[0]].tolist(), df[i[1]].tolist())
        if maxim==1 and abs(corr)>abs(result[1]) and i[0]!=i[1]:
            result = [i, corr]
        elif maxim==0 and abs(corr)<abs(result[1]) and i[0]!=i[1]:
            result =  [i,corr]
    return result

"""Initial file processing"""
#Three test files:
filepath = "mtcars.csv"
filepath2 = "div.csv"
filepath2 = "superstoreSales.csv"
filepath = "california.csv"

fin = pd.read_csv(filepath)

"""Data Integrity checks"""
df = pd.DataFrame(fin)
#Remove empty columns
empty_cols = [col for col in df.columns if df[col].isnull().all()]
# Drop these columns from the dataframe
df.drop(empty_cols,
        axis=1,
        inplace=True)

headers = list(df.columns)
headers[-1]=headers[-1].strip("\n")
boolean, numeric = [],[]
for h in headers:
    temp = df[h].dropna()
    temp = [i for i in temp if i is not None]
    
    if (len(set(temp))==2 and len(temp)>2):
        boolean.append(h)
    elif (df[h].dtype == np.int64 or df[h].dtype==np.float64) :
        numeric.append(h)
df = df.loc[:,~df.columns.duplicated()]
dup = df[df.duplicated(keep="first")]


#Processing duplicates and n
dfcopy = df.copy(deep = True)
column_means = df.interpolate()
for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
    if i in boolean:
        df[i].fillna(df[i].mode(),inplace=True)
    else:
        df[i].fillna(df[i].interpolate(method='spline', order = 3, limit_direction='forward'),inplace=True)
df = df.drop_duplicates()
dfcopy = dfcopy.drop_duplicates()
df = df.dropna()


"""Calculate correlation between numeric fields"""
pairs = list(combinations(numeric, 2))
if len(pairs)>0:
    p,corr =corrmaxmin(df,pairs,1)
    p2,corr2 = corrmaxmin(df,pairs,0)

    #notunique = minuniquepair(headers, unique)
    """Data visualisation"""
    #Graph with two fields with best corr and table with frequency of column with least uniques
    plotter(df,p, corr,1)#Call to make graph of best correlation
    
    plotter(df,p2, corr2,0)#Call to make graph of worst correlation
    
    
"""New File"""
file = "results\\"+filepath.split(".")[0]+"(checked).csv"
count = 0
while os.path.isfile(file)==True:
    count+=1
    file = "results\\"+filepath.split(".")[0]+"(" +str(count)+").csv"
df.to_csv(file)
os.startfile(file)