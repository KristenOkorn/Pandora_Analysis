# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 08:55:10 2022

@author: okorn
"""
#Import helpful toolboxes etc
import pandas as pd
import matplotlib.pyplot as plt
import os

## data loading and cleaning
#create a directory path for us to pull from / save to
path = 'C:\\Users\\okorn\\Documents\\NASA Postdoc\\Sample Pandora Data\\Long-Term Data\\'
# read in the first worksheet from the workbook myexcel.xlsx
filename = 'Wrightwood_No2.xlsx'
filepath = os.path.join(path, filename)
MVno2 = pd.read_excel(filepath, index_col=0)  
# display the first five rows in pandas to check
MVno2.head()
#rename the columns (by index) to something more convenient
MVno2.columns.values[0] = "NO2"
#get rid of columns (by index) we don't need to use
MVno2.drop(MVno2.columns[[1,2]], axis=1, inplace=True)
#remove negative column values
MVno2 = MVno2[(MVno2['NO2']>0)]

## timeseries plot (just one location)
plt.xlabel("Datetime")
plt.ylabel("Total Column NO2 (dobson)")
plt.title('Wrightwood Total Column NO2')
plt.plot(MVno2['NO2'])
imgname = 'Wrightwoodno2timeseries.png'
imgpath = os.path.join(path, imgname)
plt.savefig(imgpath)
plt.show()