#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Author: Pin-Ching Li and Ankit Ghanghas
This code generates GPM dataset from local device
The GHCN dataset is downloaded from CDO client
Graphical Analysis is applied in this code
The GPM dataset shall be downloaded in advance
"""

# read nc4 file which contains satellite data
from netCDF4 import Dataset
import numpy as np
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt         
get_ipython().run_line_magic('matplotlib', 'inline')

# the location of GPM data
folder = "C:\\Users\\Public\\Documents\\Research\\GPM_IN\\"
# start date of our dataset
f_day = date(2016,1,1)      # This is the "days since" part
# days of three years including a leap year
data_len = 365*2+366

# Empty Lists for Variables of nc4
Date = []
Time = []
PCal_IN = []
RE_IN =[]
PCal_TX = []
RE_TX =[]

# read the data by date
for d in range(data_len):
    # nc4 file name of GPM 
    f_now = str(f_day.year)+"{:02d}".format(f_day.month)+"{:02d}".format(f_day.day)
    f_name = "3B-DAY.MS.MRG.3IMERG." + f_now + "-S000000-E235959.V06.nc4"
    filename = folder + f_name
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


# # Indiana

# In[ ]:


# Get the maximum precipitation at each grid
day_max_IN = PCal_IN[0]
for i in range(len(PCal_IN)):
    day_max_IN = np.maximum(day_max_IN,PCal_IN[i])


# In[ ]:


import os
os.environ['PROJ_LIB'] = 'C:\\Users\\Pin-Ching Li\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap'
from mpl_toolkits.basemap import Basemap, cm
# define precipitation datasests
precip = day_max_IN
theLats= lats_IN
theLons= lons_IN

# Plot the figure, define the geographic bounds
fig = plt.figure(dpi=300)
latcorners = ([37,42])
loncorners = ([-89,-84])

# create basemap wit the given extent 
m = Basemap(projection='cyl',llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],llcrnrlon=loncorners[0],urcrnrlon=loncorners[1])

# Draw state and country boundaries, edge of map.
m.drawstates()
m.drawcountries()

# Draw filled contours.
clevs = np.arange(0,1.26,0.125)

# Define the latitude and longitude data
x, y = np.float32(np.meshgrid(theLons, theLats))

# Mask the values less than 0 because there is no data to plot.
masked_array = np.ma.masked_where(precip < 0,precip)

# Plot the contour of GPM in IN
cs = m.contourf(x,y,precip,alpha=0.7)

# draw the lines of x,y ticks
parallels = np.arange(37.,42.,.5)
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(-89.,-83.,.5)
m.drawmeridians(meridians,labels=[False,False,False,True])

# Set the title and fonts
plt.title('Maximum daily rainfall_Indiana \n 2016-2018')
font = {'weight' : 'bold', 'size' :4}
plt.rc('font', **font)

# Add colorbar
cbar = m.colorbar(cs,location='right',pad="5%")
cbar.set_label('mm')
# save plot
plt.savefig('MaxGPM_IN.png',dpi=200)


# # Texas

# In[ ]:


# Get the maximum precipitation at each grid
day_max_TX = PCal_TX[0]
for i in range(len(PCal_TX)):
    day_max_TX = np.maximum(day_max_TX,PCal_TX[i])


# In[ ]:


import os
os.environ['PROJ_LIB'] = 'C:\\Users\\Pin-Ching Li\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap'
from mpl_toolkits.basemap import Basemap, cm

# define precipitation datasests
precip = np.squeeze(day_max_TX)
theLats= lats_TX
theLons= lons_TX

# Plot the figure, define the geographic bounds
fig = plt.figure(dpi=300)
latcorners = ([25,38])
loncorners = ([-108,-92])

# create basemap wit the given extent 
m = Basemap(projection='cyl',llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],llcrnrlon=loncorners[0],urcrnrlon=loncorners[1])

# Draw coastlines, state and country boundaries, edge of map.
m.drawstates()
m.drawcountries()
m.drawcoastlines()

# Draw filled contours.
clevs = np.arange(0,100,5)

# Define the latitude and longitude data
x, y = np.float32(np.meshgrid(theLons, theLats))

# Mask the values less than 0 because there is no data to plot.
masked_array = np.ma.masked_where(precip < 0,precip)

# Plot the data
cs = m.contourf(x,y,precip,alpha=0.7)

# draw the lines of x,y ticks
parallels = np.arange(25.,38.,2)
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(-107.,-93.,2)
m.drawmeridians(meridians,labels=[False,False,False,True])

# Set the title and fonts
plt.title('Maximum daily rainfall_Texas \n 2016-2018')
font = {'weight' : 'bold', 'size' :5}
plt.rc('font', **font)

# Add colorbar
cbar = m.colorbar(cs,location='right',pad="5%")
cbar.set_label('mm')

# save plot
plt.savefig('MaxGPM_TX.png',dpi=200)


# # TX NCDC weather data

# In[ ]:


import numpy as np
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt         
get_ipython().run_line_magic('matplotlib', 'inline')
from cdo_api_py import Client
import datetime 
# use token to get weather file from NCDC database
# token can be requested at here: https://www.ncdc.noaa.gov/cdo-web/token
token = "<your token>"# be sure not to share your token publicly

# create client object for downloading files
my_client = Client(token, default_units="None", default_limit=1000)

#The extend is the lat, long of the target region
extent = {"north": 37, "south": 26,
          "east": -93, "west": -107}

# input of start date, end date, type of dataset
startdate = datetime.datetime(2016, 1, 1)
enddate = datetime.datetime(2018, 12, 31)
datasetid="GHCND" 

# The find_stations function returns the dataframe containing stations' info within the input extent.
stations = my_client.find_stations(
            datasetid=datasetid,
            extent=extent,
            startdate=startdate,
            enddate=enddate,
            return_dataframe=True)
# Drop the station which has less record than our criterion in the staation dataframe
dropind = []
# Drop station without enough date of observation
for i in range(len(stations.maxdate)):
    # get max and min date of each station
    date_str_max= stations.maxdate[i]
    date_str_min= stations.mindate[i]
    # transfer string to datetime
    datelast = datetime.datetime.strptime(date_str_max, '%Y-%m-%d')  
    datefirst= datetime.datetime.strptime(date_str_min, '%Y-%m-%d')
    # get the position of stations with insufficient daily data
    if datelast-enddate < datetime.timedelta(days=0):
        dropind.append(i)
    elif datefirst-startdate > datetime.timedelta(days=0):
        dropind.append(i)        
# delete stations without enough time length
stations = stations.drop(stations.index[dropind])
stations_raw= stations

# Get names of index for which datacoverage less than 0.99
indexNames = stations[ stations['datacoverage'] < 0.99 ].index
# Delete these row indexes from dataFrame
stations.drop(indexNames , inplace=True)
# Get the final station list for downloading
stations_TX = stations


# In[ ]:


# # Get data from the NCDC client
# i =0
# for rowid, station in stations_TX.iterrows(): 
#     # try downloading due to some unaccessible sites of database
#     try:
#         station_data = my_client.get_data_by_station(
#                         datasetid=datasetid,
#                         stationid=station['id'],
#                         startdate=startdate,
#                         enddate=enddate,
#                         return_dataframe=True,
#                         include_station_meta=True)
#         # set datetime to index
#         station_data.set_index(pd.to_datetime(station_data['date']), inplace =True)
#         # Drop all rows except for rainfall
#         Rainfall_day = station_data.filter(['PRCP'])
#         Rainfall_day = Rainfall_day.rename(columns={"PRCP": station['id']})
#         # Merge the dataframe of rainfall from each station
#         if i ==0:
#             merged= Rainfall_day
#         else:
#             merged =pd.merge(merged,Rainfall_day ,how='outer', left_index=True, right_index=True)
#         i +=1
#     # print the id of broken dataset
#     except:
#         print(station['id'])
# # The unit of rainfall is tenth of mm
# # Get the final rainfall dataset
# GHCN_TX  = merged


# In[ ]:


GHCN_TX = pd.read_csv('GHCN_TX.csv')


# In[ ]:


# maximum hist plots
# get describe dataframe which contains max, mean, std of dataset in each station
# do the graphical analysis among all stations
Describe_TX = GHCN_TX.describe()
# get the maximum precipitation series of all station
max_NCDC_TX = Describe_TX.iloc[7]
# set font size
plt.rcParams.update({'font.size': 12})
# plot histogram of maximum precipitation
plt.hist(max_NCDC_TX)
# label and title of plots
plt.xlabel('Maximum Precipitation (tenth of mm)')
plt.ylabel('amount of gages')
plt.title('Texas NCEI rainfall gages \n histogram of Maximum rainfall')
plt.show()
# KDE of maximum precipitation in TX with gaussian kernel
ax = max_NCDC_TX.plot.kde(bw_method=0.5)
plt.xlabel('Maximum Precipitation (tenth of mm)')
plt.title('Texas NCEI rainfall gages \n KDE plot of Maximum rainfall')


# In[ ]:


# mean hist plots
# get the mean precipitation series of all station
mean_NCDC_TX = Describe_TX.iloc[1]
# set font size
plt.rcParams.update({'font.size': 12})
# plot histogram of mean precipitation
plt.hist(mean_NCDC_TX)
# label and title of plots
plt.xlabel('Mean Precipitation (tenth of mm)')
plt.ylabel('amount of gages')
plt.title('Texas NCEI rainfall gages \n histogram of Mean rainfall')
plt.show()
# KDE of mean precipitation in TX with gaussian kernel
ax = mean_NCDC_TX.plot.kde(bw_method=0.5)
plt.xlabel('Mean Precipitation (tenth of mm)')
plt.title('Texas NCEI rainfall gages \n KDE plot of Mean rainfall')


# In[ ]:


# std hist plots
# get the std precipitation series of all station
std_NCDC_TX = Describe_TX.iloc[2]
# set font size
plt.rcParams.update({'font.size': 12})
# plot histogram of precipitation std
plt.hist(std_NCDC_TX)
# label and title of plot
plt.xlabel('std Precipitation (tenth of mm)')
plt.ylabel('amount of gages')
plt.title('Texas NCEI rainfall gages \n histogram of rainfall std')
plt.show()
# KDE of precipitation std in TX with gaussian kernel
ax = std_NCDC_TX.plot.kde(bw_method=0.5)
plt.xlabel('std Precipitation (tenth of mm)')
plt.title('Texas NCEI rainfall gages \n KDE plot of rainfall std')


# In[ ]:


import os
os.environ['PROJ_LIB'] = 'C:\\Users\\Pin-Ching Li\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap'
from mpl_toolkits.basemap import Basemap, cm
# Plot the figure, define the geographic bounds
fig = plt.figure(dpi=300)
latcorners = ([25,38])
loncorners = ([-108,-92])

# create basemap wit the given extent 
m = Basemap(projection='cyl',llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],llcrnrlon=loncorners[0],urcrnrlon=loncorners[1])

# Draw coastlines, state and country boundaries, edge of map.
m.drawstates()
m.drawcountries()
m.drawcoastlines()

# draw location of stations on basemap
m.scatter(stations_TX.longitude,stations_TX.latitude, s = 10,color = 'red')

# draw the lines of x,y ticks
parallels = np.arange(25.,38.,2)
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(-108.,-92.,2)
m.drawmeridians(meridians,labels=[False,False,False,True])

# Set the title and fonts
plt.title('GHCN Daily Rainfall in Texas \n 2016-2018')
font = {'weight' : 'bold', 'size' :4}
plt.rc('font', **font)
# save plot
plt.savefig('Stations_TX.png',dpi=200)


# # IN NCDC weather data

# In[ ]:


import numpy as np
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt         
get_ipython().run_line_magic('matplotlib', 'inline')
from cdo_api_py import Client
import datetime 
# use token to get weather file from NCDC database
# token can be requested at here: https://www.ncdc.noaa.gov/cdo-web/token
token = "<your token>"# be sure not to share your token publicly

my_client = Client(token, default_units="None", default_limit=1000)
#The extend is the lat, long of the target region: obtain from boundary.shp.
extent = {"north": 41.76, "south": 37.8,
          "east": -84.81, "west": -88.03}
# input of start date, end date, type of dataset, and name of gauge
startdate = datetime.datetime(2016, 1, 1)
enddate = datetime.datetime(2018, 12, 31)
datasetid="GHCND" 
#The find_stations function returns the dataframe containing stations' info within the input extent.
stations = my_client.find_stations(
            datasetid=datasetid,
            extent=extent,
            startdate=startdate,
            enddate=enddate,
            return_dataframe=True)
dropind = []
# Drop station without enough date of observation
for i in range(len(stations.maxdate)):
    # get max and min date of each station
    date_str = stations.maxdate[i]
    date_str_min= stations.mindate[i]
    # transfer string to datetime
    datelast = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    datefirst= datetime.datetime.strptime(date_str_min, '%Y-%m-%d')
    # get the position of stations with insufficient daily data
    if datelast-enddate < datetime.timedelta(days=0):
        dropind.append(i)
    elif datefirst-startdate > datetime.timedelta(days=0):
        dropind.append(i)        
# delete stations without enough time length
stations = stations.drop(stations.index[dropind])
stations_raw= stations

# Get names of indexes for which datacoverage less than 0.95
indexNames = stations[ stations['datacoverage'] < 0.95 ].index
# Delete these row indexes from dataFrame
stations.drop(indexNames , inplace=True)

# other gages with available data less than 0.95
Insuff_gage = ['GHCND:US1INBN0010', 'GHCND:US1INBW0010',
                 'GHCND:US1INCW0003', 'GHCND:US1INLK0046',
                 'GHCND:US1INMN0007', 'GHCND:US1INMT0001',
                 'GHCND:US1INPS0001', 'GHCND:US1KYBT0001',
                 'GHCND:USC00116558', 'GHCND:USC00121873',
                 'GHCND:USC00122041', 'GHCND:USC00124244',
                 'GHCND:USC00126801']
# get rid of the stations in list above
# Get the final station list for downloading
stations_IN = stations[~stations['id'].isin(Insuff_gage)]


# In[ ]:


# # Get data from the NCDC client
# i = 0
# for rowid, station in stations_IN.iterrows(): 
#     # try downloading due to some unaccessible sites of database
#     try:
#         station_data = my_client.get_data_by_station(
#                         datasetid=datasetid,
#                         stationid=station['id'],
#                         startdate=startdate,
#                         enddate=enddate,
#                         return_dataframe=True,
#                         include_station_meta=True)
#         # set datetime to index
#         station_data.set_index(pd.to_datetime(station_data['date']), inplace =True)
#         # Drop all rows except for rainfall
#         Rainfall_day = station_data.filter(['PRCP'])
#         Rainfall_day = Rainfall_day.rename(columns={"PRCP": station['id']})
#         # Merge the dataframe of rainfall from each station
#         if i ==0:
#             merged= Rainfall_day
#         else:
#             merged =pd.merge(merged,Rainfall_day ,how='outer', left_index=True, right_index=True)
#         i +=1
#     # print the id of broken dataset
#     except:
#         print(station['id'])
# # The unit of rainfall is tenth of mm
# # Get the final rainfall dataset
# GHCN_IN  = merged


# In[ ]:


GHCN_IN = pd.read_csv('GHCN_IN.csv')


# In[ ]:


# maximum hist plots
# get describe dataframe which contains max, mean, std of dataset in each station
# do the graphical analysis among all stations
Describe_IN = GHCN_IN.describe()
# get the maximum precipitation series of all station
max_NCDC = Describe_IN.iloc[7]
# set font size
plt.rcParams.update({'font.size': 12})
# plot histogram of maximum precipitation
plt.hist(max_NCDC)
# label and title of plots
plt.xlabel('Maximum Precipitation (tenth of mm)')
plt.ylabel('amount of gages')
plt.title('Indiana NCEI rainfall gages \n histogram of Maximum rainfall')
plt.show()
# KDE of maximum precipitation in IN with gaussian kernel
ax = max_NCDC.plot.kde(bw_method=0.5)
plt.xlabel('Maximum Precipitation (tenth of mm)')
plt.title('Indiana NCEI rainfall gages \n KDE plot of Maximum rainfall')


# In[ ]:


# mean precipitation hist plots
# get the mean precipitation series of all station
mean_NCDC = Describe_IN.iloc[1]
# set font size
plt.rcParams.update({'font.size': 12})
# plot histogram of mean precipitation
plt.hist(mean_NCDC)
# label and title of plots
plt.xlabel('Mean Precipitation (tenth of mm)')
plt.ylabel('amount of gages')
plt.title('Indiana NCEI rainfall gages \n histogram of Mean rainfall')
plt.show()
# KDE of mean precipitation in TX with gaussian kernel
ax = mean_NCDC.plot.kde(bw_method=0.5)
plt.xlabel('Minimum Precipitation (tenth of mm)')
plt.title('Indiana NCEI rainfall gages \n KDE plot of Mean rainfall')


# In[ ]:


# std of precipitation hist plots
# get the precipitation std series of all station
std_NCDC = Describe_IN.iloc[2]
# set font size
plt.rcParams.update({'font.size': 12})
# plot histogram of precipitation std
plt.hist(std_NCDC)
# label and title of plots
plt.xlabel('std of Precipitation (tenth of mm)')
plt.ylabel('amount of gages')
plt.title('Indiana NCEI rainfall gages \n histogram of rainfall std')
plt.show()
# KDE of precipitation std in IN with gaussian kernel
ax = std_NCDC.plot.kde(bw_method=0.5)
plt.xlabel('std of Precipitation (tenth of mm)')
plt.title('Indiana NCEI rainfall gages \n KDE plot of rainfall std')


# In[ ]:


# Box plot of varialbes across the states
# make axis containing state names
ax = plt.axes()
# draw boxplot with whisker = 1.5IQR
plt.boxplot([max_NCDC,max_NCDC_TX])
# define font size
plt.rcParams.update({'font.size': 12})
# label and title
plt.ylabel('Maximum Precipitation (tenth of mm)')
plt.title('NCEI rainfall gages \n Boxplot of Maximum rainfall')
# set state name as x label
ax.set_xticklabels(['Indiana', 'Texas'])
plt.show()


# In[ ]:


# Box plot
# make axis containing state names
ax = plt.axes()
# draw boxplot with whisker = 1.5IQR
plt.boxplot([mean_NCDC,mean_NCDC_TX])
# define font size
plt.rcParams.update({'font.size': 12})
# label and title
plt.ylabel('Mean Precipitation (tenth of mm)')
plt.title('NCEI rainfall gages \n Boxplot of Mean rainfall')
# set state name as x label
ax.set_xticklabels(['Indiana', 'Texas'])
plt.show()


# In[ ]:


# Box plot
# make axis containing state names
ax = plt.axes()
# draw boxplot with whisker = 1.5IQR
plt.boxplot([std_NCDC,std_NCDC_TX])
# define font size
plt.rcParams.update({'font.size': 12})
# label and title
plt.ylabel('std Precipitation (tenth of mm)')
plt.title('NCEI rainfall gages \n Boxplot of rainfall std')
# set state name as x label
ax.set_xticklabels(['Indiana', 'Texas'])
plt.show()


# In[ ]:


import os
os.environ['PROJ_LIB'] = 'C:\\Users\\Pin-Ching Li\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap'
from mpl_toolkits.basemap import Basemap, cm
# Plot the figure, define the geographic bounds
fig = plt.figure(dpi=300)
latcorners = ([37,42])
loncorners = ([-89,-84])
# create basemap wit the given extent 
m = Basemap(projection='cyl',llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],llcrnrlon=loncorners[0],urcrnrlon=loncorners[1])

# Draw state and country boundaries, edge of map.
m.drawstates()
m.drawcountries()
# Draw loaction of stations on base map
m.scatter(stations_IN.longitude,stations_IN.latitude, s = 10,color = 'red')
# Draw lines of x,y ticks
parallels = np.arange(37.,42.,.5)
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(-89.,-83.,.5)
m.drawmeridians(meridians,labels=[False,False,False,True])

# Set the title and fonts
plt.title('GHCN Daily Rainfall in Indiana \n 2016-2018')
font = {'weight' : 'bold', 'size' :4}
plt.rc('font', **font)
# Save the plot
plt.savefig('Stations_IN.png',dpi=200)


# In[ ]:


"""
Spatial distribution of stations with max precip larger than /
whisker end of box plot (Q3+1.5IQR)
"""
# get the describe of decribe dataframe 
# (maximum series according to stations)
Max_Des_TX = Describe_TX.iloc[7].describe()
# get IQR by Q3-Q1
IQR_TX = Max_Des_TX[6] - Max_Des_TX[4]
# get whisker upper bound by Q3+1.5IQR
whisbound = Max_Des_TX[6]+IQR_TX*1.5
# get stations with max precipitation larger than whiskers
Largerlabel_TX = []
for i in range(Describe_TX.shape[1]):
    if Describe_TX.iloc[7][i] > whisbound:
        Largerlabel_TX.append(Describe_TX.columns[i])
LargerStation_TX = stations_TX[stations_TX['id'].isin(Largerlabel_TX)]


# In[ ]:


import os
os.environ['PROJ_LIB'] = 'C:\\Users\\Pin-Ching Li\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap'
from mpl_toolkits.basemap import Basemap, cm
# Plot the figure, define the geographic bounds
fig = plt.figure(dpi=300)
latcorners = ([25,38])
loncorners = ([-108,-92])
# Create basemap
m = Basemap(projection='cyl',llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],llcrnrlon=loncorners[0],urcrnrlon=loncorners[1])

# Draw coastlines, state and country boundaries, edge of map.
m.drawstates()
m.drawcountries()
m.drawcoastlines()
# Draw stations on basemap
m.scatter(LargerStation_TX.longitude,LargerStation_TX.latitude, s = 10,color = 'red')
# Draw lines of x,y ticks
parallels = np.arange(25.,38.,2)
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(-108.,-92.,2)
m.drawmeridians(meridians,labels=[False,False,False,True])
# Set the title and fonts
plt.title('GHCN Daily Rainfall larger than whisker in Texas \n 2016-2018')
font = {'weight' : 'bold', 'size' :5}
plt.rc('font', **font)
# Save figure
plt.savefig('LargerMax_TX.png',dpi=200)


# In[ ]:


"""
Spatial distribution of stations with max precip larger than /
whisker end of box plot (Q3+1.5IQR)
"""
# get the describe of decribe dataframe 
# (std series according to stations)
std_Des_TX = Describe_TX.iloc[2].describe()
# get IQR by Q3-Q1
IQR_TX = std_Des_TX[6] - std_Des_TX[4]
# get whisker upper bound by Q3+1.5IQR
whisbound = std_Des_TX[6]+IQR_TX*1.5
# get stations with precipitation std larger than whiskers upped bound
Largerlabel_TX = []
for i in range(Describe_TX.shape[1]):
    if Describe_TX.iloc[2][i] > whisbound:
        Largerlabel_TX.append(Describe_TX.columns[i])
LargerStation_TX = stations_TX[stations_TX['id'].isin(Largerlabel_TX)]


# In[ ]:


import os
os.environ['PROJ_LIB'] = 'C:\\Users\\Pin-Ching Li\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap'
from mpl_toolkits.basemap import Basemap, cm
# Plot the figure, define the geographic bounds
fig = plt.figure(dpi=300)
latcorners = ([25,38])
loncorners = ([-108,-92])
# Create basemap
m = Basemap(projection='cyl',llcrnrlat=latcorners[0],urcrnrlat=latcorners[1],llcrnrlon=loncorners[0],urcrnrlon=loncorners[1])

# Draw coastlines, state and country boundaries, edge of map.
m.drawstates()
m.drawcountries()
m.drawcoastlines()
# Draw stations on basemap
m.scatter(LargerStation_TX.longitude,LargerStation_TX.latitude, s = 10,color = 'red')
# Draw lines of x,y ticks
parallels = np.arange(25.,38.,2)
m.drawparallels(parallels,labels=[True,False,True,False])
meridians = np.arange(-108.,-92.,2)
m.drawmeridians(meridians,labels=[False,False,False,True])

# Set the title and fonts
plt.title('GHCN Daily Rainfall with large std in Texas \n 2016-2018')
font = {'weight' : 'bold', 'size' :5}
plt.rc('font', **font)
# Save figure
plt.savefig('Largerstd_TX.png',dpi=200)


# In[ ]:


"""
PPplot for maximum precipitation
Fit the spatial distributed data with normal distribution
"""
import statsmodels.api as sm
import scipy.stats as  stats 
# Get probability plot of IN and TX
pplot = sm.ProbPlot(Describe_IN.iloc[7],dist=stats.norm,fit=True)
pplot2 = sm.ProbPlot(Describe_TX.iloc[7],dist=stats.norm,fit=True)
# Draw PP Plot of them with 45 degree line: normal distribution
pplot.ppplot(line='45')
# setup font size
plt.rcParams.update({'font.size': 12})
plt.title('PP-plot of Maximum Precipitation \n Indiana')
plt.show()

# Draw PP Plot of them with 45 degree line: normal distribution
pplot2.ppplot(line='45')
# setup font size
plt.rcParams.update({'font.size': 12})
plt.title('PP-plot of Maximum Precipitation \n Texas')
plt.show()

