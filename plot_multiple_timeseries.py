# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:56:27 2022

@author: okorn
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:55:10 2022

@author: okorn
"""
#Import helpful toolboxes etc
import pandas as pd
import matplotlib.pyplot as plt
import os

#create a directory path for us to pull from / save to
path = 'C:\\Users\\okorn\\Documents\\NASA Postdoc\\Sample Pandora Data\\Long-Term Data\\'

#plotting to initialize
plt.xlabel("Datetime")
plt.ylabel("Total Column NO2 (dobson)")
plt.title('California Pandora Total Column NO2')

#set up structures to loop through
locations = ['MV','Richmond','SanJose','Wrightwood']
#loop through for each location
for i in range(4):
    #get the location alone
    location = locations[i]
    #add it to the string to be loaded
    filename = "{}_NO2.xlsx".format(location)
    # read in the first worksheet from the workbook myexcel.xlsx
    filepath = os.path.join(path, filename)
    no2 = pd.read_excel(filepath, index_col=0)  
    #rename the columns (by index) to something more convenient
    no2.columns.values[0] = "NO2"
    #get rid of columns (by index) we don't need to use
    no2.drop(no2.columns[[1,2]], axis=1, inplace=True)
    #remove negative column values
    no2 = no2[(no2['NO2']>0)]
    #timeseries plot (will layer each location)
    plt.plot(no2['NO2'], label = locations[i])
    plt.legend()

#final plotting & saving
imgname = 'NO2timeseries.png'
imgpath = os.path.join(path, imgname)
plt.savefig(imgpath)
plt.show()