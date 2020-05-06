# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:21:59 2020

Author: Pin-Ching Li and Ankit Ghanghas

This Script performs the basic data quality checks on precipitation measurements and the user needs to also specify the 
state to which the data belongs and the type of data being processed where it is 'GPM' or 'GHCN'

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def ReadData(filename):
    """

    Parameters
    ----------
    filename : This is the input filename

    Returns
    -------
    DataDF : The output pandas dataframe of the raw data in the filename file. This dataframe has 
    year, month and day as index and the column names represent the Ground Station ID for that station.
    
    ReplacedDF : This is the dicitonary dataframe designed to contain all the missing value counts

    """
    # open and read the file
    DataDF = pd.read_csv(filename)
    DataDF = DataDF.set_index('date')
    
    colnames = DataDF.columns.values # gets the names of all the column names in the input file. These are the station IDs
    
    # define and initialize the missing data dictionary
    ReplacedDF = pd.DataFrame(0, index=["1. No Data"], columns=colnames)
    return (DataDF, ReplacedDF)


def Check_RemoveNOData (DataDF, ReplacedDF):
    """
    Parameters
    ----------
    DataDF : Input Raw dataframe with values.
    ReplacedDF : Disctionary dataframe containing count of missing values

    Returns
    -------
    DataDF : Corrected Dataframe with replaced no data values represented by -9999.9 or empty by nan 
    ReplacedDF : Counts the number of missing values

    """
    DataDF[DataDF == -9999.9]=np.nan
    ReplacedDF.loc['1. No Data',:]=DataDF.isna().sum()
    return (DataDF, ReplacedDF)

def Check_grossErrors(DataDF, ReplacedDF, State):
    """
    This function check for gross errors, values well outside the expected range and removes them from the dataset.
    the function returns the modified dataframes with the data that has passed and counts the data that failed the test
    

    Parameters
    ----------
    State : This indicates the state used for data quality checking if We work on Texas then max allowable precipitaiton value is 800 mm for a day
    this is so high for texas because of Hurricane Harvey That occured in the time considered.
    if the data belongs to Indiana, max allowed precipitaiton is 250 mm

    """
    
    
    if State == 'TX':
        a= 8000
    else:
        a= 2500
    DataDF[DataDF<0]=np.nan
    DataDF[DataDF>a]=np.nan
    ReplacedDF.loc['2. Gross Error',: ]= DataDF.isna().sum() - ReplacedDF.sum()
    
    return (DataDF, ReplacedDF)


def savetocsv(filename,DataDF,ReplacedDF):
    """ 
    This function takes in the filename and saves in the modified input datframe and dictionary dataframe as csv file named 
    'file_corrected.csv' and 'file_data_quality_metric.csv' respectively
    """
    filename=filename[:-4]
    DataDF.to_csv((filename+"_corrected.csv"))
    ReplacedDF.to_csv((filename+"_data_quality_metric.csv"))
    
def main(filename,stateshortname, source):
    """
    This is the main function which combines all the funcitons mentioned above and publishes images
    of data quality if there are any erros in the data.
    
    #since individual observations show that the used dataset do not have any gross Errors for the given time frame
    so this function plots only the places with NO data values if any stations with missing values are found.
    
    The function takes three input values, "filename": name of the file which contains the data
    "statesshortname" : this specifies the State in US for which the data data quality checking is done
    "source": This specifies the source of input data, whether it is GPM or GHCN.
    """
    
    file=filename
    state=stateshortname
    
    df, replace = ReadData(file)
    df,replace=Check_RemoveNOData(df, replace)
    df, replace= Check_grossErrors(df, replace, state)
    savetocsv(file, df, replace)
    a=replace.loc['1. No Data',:] # extracts the no data row from dictionary
    if source=='GPM' or source=='GHCN':
        if len(a[a>0].index.values)>0:
            if state=='IN':    
                stations=pd.read_csv('stations_IN.csv')
                stations=stations.set_index('id')
                plt.figure(dpi=300)
                # Draw stations on basemap
                plt.scatter(stations.longitude[a[a>0].index.values],stations.latitude[a[a>0].index.values], s = a[a>0],color = 'red')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title('Indiana Stations with Missing Data Values')
                if source=='GHCN':
                    plt.savefig('IN_GHCN_nodata.png', dpi=200)
                elif source=='GPM' :
                    plt.savefig('IN_GPM_nodata.png', dpi=200)
                
            elif state=='TX':
                stations=pd.read_csv('stations_TX.csv')
                stations=stations.set_index('id')
                plt.figure(dpi=300)
                # Draw stations on basemap
                plt.scatter(stations.longitude[a[a>0].index.values],stations.latitude[a[a>0].index.values], s = a[a>0],color = 'red')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title('Texas Stations with Missing Data Values')
                if source=='GHCN':
                    plt.savefig('TX_GHCN_nodata.png', dpi=200)
                elif source=='GPM' :
                    plt.savefig('TX_GPM_nodata.png', dpi=200)

            
        
        

main('GHCN_IN.csv','IN','GHCN')
main('GHCN_TX.csv','TX','GHCN')
main('GPM_IN_avg.csv','IN','GPM')
main('GPM_IN_max.csv','IN','GPM')
main('GPM_TX_avg.csv','IN','GPM')
main('GPM_TX_max.csv','IN','GPM')
