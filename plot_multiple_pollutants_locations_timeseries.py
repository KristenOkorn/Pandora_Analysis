# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:03:02 2022

@author: okorn
"""

#Import helpful toolboxes etc
import pandas as pd
import matplotlib.pyplot as plt
import os

#create a directory path for us to pull from / save to
path = 'C:\\Users\\okorn\\Documents\\NASA Postdoc\\Sample Pandora Data\\Long-Term Data\\'

#pollutant loop first

#set up pollutant structures to loop through
pollutants = ['NO2','SO2','O3']

#loop through for each pollutant
for n in range(3):
    #get the pollutant name alone
    pollutant = pollutants[n]
    
    #plotting to initialize
    plt.xlabel("Datetime")
    #add string + variable for y axis
    y = 'Total Column {} (dobson)'.format(pollutant)
    plt.ylabel(y)
    #add string + variable for plot title
    titl = 'California Pandora Hourly Total Column {}'.format(pollutant)
    plt.title(titl)

    #set up location structures to loop through
    locations = ['MV','Richmond','SanJose','Wrightwood']
    #loop through for each location
    for i in range(4):
        #get the location alone
        location = locations[i]
        #add it to the string to be loaded
        filename = "{}_{}.xlsx".format(location,pollutant)
        # read in the first worksheet from the workbook myexcel.xlsx
        filepath = os.path.join(path, filename)
        compound = pd.read_excel(filepath, index_col=0)  
        #rename the columns (by index) to something more convenient
        compound.columns.values[0] = "compound"
        #get rid of columns (by index) we don't need to use
        compound.drop(compound.columns[[1,2]], axis=1, inplace=True)
        #remove negative column values
        compound = compound[(compound['compound']>0)]
        #retime the data to hourly mean
        compound = compound.resample('H').mean()
        #timeseries plot (will layer each location)
        plt.plot(compound['compound'], label = locations[i])
        plt.legend()

    #final plotting & saving
    imgname = '{}hourlytimeseries.png'.format(pollutant)
    imgpath = os.path.join(path, imgname)
    plt.savefig(imgpath)
    plt.show()