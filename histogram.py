# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:09:22 2022

@author: okorn
"""

#Import helpful toolboxes etc
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

#create a directory path for us to pull from / save to
path = 'C:\\Users\\okorn\\Documents\\NASA Postdoc\\Sample Pandora Data\\Long-Term Data\\'


#pollutant loop first

#set up pollutant structures to loop through
pollutants = ['NO2','SO2','O3']

#loop through for each pollutant
for n in range(3):
    #get the pollutant name alone
    pollutant = pollutants[n]
    
    #initialize a dataframe to add the concentrations of each to
    newdata = pd.DataFrame()

    #set up location structures to loop through
    locations = ['MountainView','Richmond','SanJose','Wrightwood']
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
        compound.columns.values[0] = 'compound'
        #compound.columns.values[0] = "{}".format(location)
        #get rid of columns (by index) we don't need to use
        compound.drop(compound.columns[[1,2]], axis=1, inplace=True)
        #remove negative column values
        compound = compound[(compound['compound']>0)]
        #retime the data to hourly mean
        compound = compound.resample('H').mean()
        #remove missing values from the dataframe
        compound = compound.dropna() 
        #create an array with the location repeating
        place=np.repeat(location, len(compound))
        #add the array to the individual dataframe
        compound['Location'] = place.tolist()
        #add this location to the overall dataframe
        newdata = newdata.append(compound,ignore_index=True)
    
    #Change dataframe to array before plotting
    #newdata = newdata.to_numpy()
    #rename columns before plotting
    #newdata = rfn.rename_fields(newdata, {0: 'compound', 1: 'Location'})
    
    #Creating axes & histogram
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    sns.set_style("whitegrid")
    ax = sns.histplot(newdata, x="compound", hue="Location", element="step")
    plt.xlim(0, 10)

    #add string + variable for y axis, plus limits
    xlabel = 'Total Column {} (dobson)'.format(pollutant)
    plt.xlabel(xlabel)

    #add string + variable for plot title
    titl = 'California Pandora Hourly Total Column {}'.format(pollutant)
    plt.title(titl)
   
    #final plotting & saving
    imgname = '{}hourlyhistogram_10.png'.format(pollutant)
    imgpath = os.path.join(path, imgname)
    plt.savefig(imgpath)
    plt.show()