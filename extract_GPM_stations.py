# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:06:35 2020

Author: Pin-Ching Li and Ankit Ghanghas


This script extracts satellite precipitation for ground stations with known lat and long.
Ulimately two datasets are created one by taking average of the four nearest grids to the 
station and one by taking max of the grids
"""

from netCDF4 import Dataset
import numpy as np
from datetime import date, timedelta
import pandas as pd     


# the location of GPM data
folder = r"C:\Users\aghangha\OneDrive - purdue.edu\coursework\notes\abe651\GPM_IN"
# start date of our dataset
f_day = date(2016,1,1)      # This is the "days since" part
# days of three years including a leap year
data_len = 365*2+366

Date = []
Time = []
PCal_IN = []
RE_IN =[]
PCal_TX = []
RE_TX =[]

stations_IN=pd.read_csv('stations_IN.csv')
stations_TX=pd.read_csv('stations_TX.csv')

GPM_IN_max=pd.DataFrame(0, index= np.arange(1096), columns=stations_IN.id.values)
GPM_TX_max=pd.DataFrame(0, index= np.arange(1096), columns=stations_TX.id.values)

GPM_IN_avg=pd.DataFrame(0, index= np.arange(1096), columns=stations_IN.id.values)
GPM_TX_avg=pd.DataFrame(0, index= np.arange(1096), columns=stations_TX.id.values)

lon_IN_station=stations_IN.longitude
lat_IN_station=stations_IN.latitude

lon_TX_station=stations_TX.longitude
lat_TX_station=stations_TX.latitude

idx=0

# read the data by date
for d in range(data_len):
    # nc4 file name of GPM 
    f_now = str(f_day.year)+"{:02d}".format(f_day.month)+"{:02d}".format(f_day.day)
    f_name = "3B-DAY.MS.MRG.3IMERG." + f_now + "-S000000-E235959.V06.nc4"
    filename = folder + "\\" + f_name
    # read GPM file using Dataset in netCDF4
    fh = Dataset(filename, mode='r')

    # Lat, long for IN and TX
    lons_IN = fh.variables['lon'][917:953].data
    lats_IN = fh.variables['lat'][1277:1318].data
    lons_TX = fh.variables['lon'][729:870].data
    lats_TX = fh.variables['lat'][1159:1270].data
    
    # days since 1970-01-01 00:00:00Z
    t    = fh.variables['time'][0].data
    
    # Three major variables are used in this study
    # Two kind of Precipitation datasets: HQ Precipitation and Cal precipitation 
    x2 = fh.variables['precipitationCal'][:]
    # Random Error of GPM
    x3 = fh.variables['randomError'][:]
    fh.close()
    # extract data from mask array stored in nc4 file
    xarray2= x2[0].data
    xarray3= x3[0].data

    # we clip the GPM of IN and TX by giving lat, long extent
    # Indiana
    Precip2_seed=[]
    randerror_seed=[]
    # set the range for clipping GPM
    
    for i in range(len(lon_IN_station)):
        lat=int((lat_IN_station.iloc[i]-(-89.95))/0.1)
        lon=int((lon_IN_station.iloc[i]-(-179.95))/0.1)
        a=np.array([xarray2[lon,lat],xarray2[lon+1,lat],xarray2[lon,lat+1],xarray2[lon+1,lat+1]])        
        GPM_IN_max.iloc[idx,i]= a.max()
        GPM_IN_avg.iloc[idx,i]= a.mean()
        
        
    for i in range(917,953):
        Precip2_seed.append(xarray2[i][1277:1318])
        randerror_seed.append(xarray3[i][1277:1318])
    # transpose the array 
    pm_IN  = np.asarray(Precip2_seed).transpose()
    re_IN  = np.asarray(randerror_seed).transpose()

    # Store variables into a list
    PCal_IN.append(pm_IN)
    RE_IN.append(re_IN)

    # Texas
    Precip2_seed=[]
    randerror_seed=[]
    # set the range for clipping GPM
    
    for i in range(len(lon_TX_station)):
        lat=int((lat_TX_station.iloc[i]-(-89.95))/0.1)
        lon=int((lon_TX_station.iloc[i]-(-179.95))/0.1)
        a=np.array([xarray2[lon,lat],xarray2[lon+1,lat],xarray2[lon,lat+1],xarray2[lon+1,lat+1]])        
        GPM_TX_max.iloc[idx,i]= a.max()
        GPM_TX_avg.iloc[idx,i]= a.mean()
        
    
    for i in range(729,870):
        Precip2_seed.append(xarray2[i][1159:1270])
        randerror_seed.append(xarray3[i][1159:1270])
    pm_TX  = np.asarray(Precip2_seed).transpose()
    re_TX  = np.asarray(randerror_seed).transpose()    
    
    # Store variables into a list
    PCal_TX.append(pm_TX)
    RE_TX.append(re_TX)
    Time.append(t)
    
    # show date of files to make sure nothing go wrong
    print(f_day)
    # add one day to current date (update filename)
    f_day = f_day + timedelta(days=1)
    idx +=1

GHCN_IN=pd.read_csv('GHCN_IN.csv')
GHCN_IN=GHCN_IN.set_index('date')
GPM_TX_max=GPM_TX_max.set_index(GHCN_IN.index) # set index to dates similar to that in GHCN dataset
GPM_TX_avg=GPM_TX_avg.set_index(GHCN_IN.index)

GPM_IN_avg=GPM_IN_avg.set_index(GHCN_IN.index)
GPM_IN_max=GPM_IN_max.set_index(GHCN_IN.index)

GPM_IN_max.to_csv('GPM_IN_max.csv')
GPM_TX_max.to_csv('GPM_TX_max.csv')

GPM_IN_avg.to_csv('GPM_IN_avg.csv')
GPM_TX_avg.to_csv('GPM_TX_avg.csv')
