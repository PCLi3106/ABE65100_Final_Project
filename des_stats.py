# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:21:14 2020

Author: Pin-Ching Li and Ankit Ghanghas

This script finds basic descriptive monthly and annual statistics for the input dataset and saves them.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats


def ReadData(filename):
    """

    Parameters
    ----------
    filename : This is the input filename

    Returns
    -------
    DataDF : The output pandas dataframe of the raw data in the filename file. This dataframe has 
    year, month and day as index and the column names represent the Ground Station ID for that station.
    

    """
    # open and read the file
    DataDF = pd.read_csv(filename)
    DataDF = DataDF.set_index('date')
    DataDF= DataDF.set_index(pd.to_datetime(DataDF.index)) # converts index to datetime index
    
    return (DataDF)

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of precipitation , after filtering out NoData
       values.  Tqmean is the fraction of time that daily rainfall
       exceeds mean rainfall for each year. The routine returns
       the Tqmean value for the given data array."""
    
    Qvalues=Qvalues.dropna()
    
    Tqmean=(Qvalues>Qvalues.mean()).sum()/len(Qvalues) 
    return ( Tqmean )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with rainfall greater 
       than 3 times the annual median rainfall. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with rain greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual rainfall value for the given data array."""
    Qvalues=Qvalues.dropna()
    median3x = (Qvalues > ((Qvalues.median())*3)).sum()    
    return ( median3x )

def GetAnnualStatistics_metric(DataDF):
    
    col_name=['Mean Rain', 'Peak Rain', 'Median Rain', 'Coeff Var', 'Skew', 'Tqmean', '3xMedian' ]
    Annual_metric=pd.DataFrame(0,index=DataDF.columns.values, columns=col_name)
    Annual_metric['Mean Rain']=DataDF.resample('Y').mean().mean()
    Annual_metric['Peak Rain']= DataDF.resample('Y').max().mean()
    Annual_metric['Median Rain']= DataDF.resample('Y').median().mean()
    Annual_metric['Coeff Var']= ((DataDF.resample('Y').std().mean())/(DataDF.resample('Y').mean().mean()))*100
    for i in range (len(DataDF.columns.values)):
        Annual_metric.iloc[i,4]=DataDF.resample('Y').apply({ DataDF.columns[i] : lambda x: stats.skew(x, nan_policy='omit',bias=False)}, raw=True).mean().values
        Annual_metric.iloc[i,5]=DataDF.resample('Y').apply({ DataDF.columns[i]: lambda x: CalcTqmean(x)}).mean().values
        Annual_metric.iloc[i,6]=DataDF.resample('Y').apply({ DataDF.columns[i]: lambda x: CalcExceed3TimesMedian(x)}).mean().values
    return Annual_metric


def GetMonthlyStatistics_metric(DataDF):
    
    col_name=['Mean Rain', 'Peak Rain', 'Coeff Var','Month']
    Monthly_metric=pd.DataFrame(0,index=DataDF.columns.values, columns=col_name)
      
    Monthly_metric['Mean Rain']=DataDF.resample('M').mean()[0::12].mean()
    Monthly_metric['Peak Rain']=DataDF.resample('M').max()[0::12].mean()
    Monthly_metric['Coeff Var']= ((DataDF.resample('M').std()[0::12].mean())/(DataDF.resample('M').mean()[0::12].mean()))*100
    Monthly_metric['Month']=1
    for i in range(1,12):
        Monthly_metric_1=pd.DataFrame(0,index=DataDF.columns.values, columns=col_name)
        Monthly_metric_1['Mean Rain']=DataDF.resample('M').mean()[i::12].mean()
        Monthly_metric_1['Peak Rain']=DataDF.resample('M').max()[i::12].mean()
        Monthly_metric_1['Coeff Var']= ((DataDF.resample('M').std()[i::12].mean())/(DataDF.resample('M').mean()[0::12].mean()))*100
        Monthly_metric_1['Month']=i+1
        Monthly_metric=Monthly_metric.append(Monthly_metric_1)
    
    return Monthly_metric

def savetocsv(filename,Annual,Monthly):
    """ 
    This function takes in the filename and saves in the modified input datframe and dictionary dataframe as csv file named 
    'file_corrected.csv' and 'file_data_quality_metric.csv' respectively
    """
    filename=filename[:-4]
    Annual.to_csv((filename+"_annual_metric.csv"))
    Monthly.to_csv((filename+"_monthly_metric.csv"))
    

def main_code(filename):
    
    data=ReadData(filename)
    annual=GetAnnualStatistics_metric(data)
    monthly=GetMonthlyStatistics_metric(data)
    savetocsv(filename, annual, monthly)
    
main_code('GHCN_IN.csv')
main_code('GHCN_TX.csv')
main_code('GPM_IN_avg.csv')
main_code('GPM_IN_max.csv')
main_code('GPM_TX_avg.csv')
main_code('GPM_TX_max.csv')
    