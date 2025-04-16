# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 03:07:27 2024

Applys each grid model to each matching surface-Pandora pair

@author: okorn
"""

#Import helpful toolboxes etc
import pandas as pd
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from joblib import load
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import multiprocessing

#list of pollutants to model
pollutants = ['O3']

#was bagging used?
bagging = 'no'

for n in range(len(pollutants)):
    
    if pollutants[n] == 'O3':
        #list of pods to model
        locations = ['Bayonne','Bristol','CapeElizabeth','Cornwall','EastProvidence','Londonderry','Lynn','MadisonCT','NewBrunswick','NewHaven','OldField','Philadelphia','Pittsburgh','Queens','WashingtonDC','Westport','SLC','AldineTX','LibertyTX','HoustonTX']
        pods = ['EPA_Bayonne','EPA_Bristol','EPA_CapeElizabeth','EPA_Cornwall','EPA_EastProvidence','EPA_Londonderry','EPA_Lynn','EPA_MadisonCT','EPA_NewBrunswick','EPA_NewHaven','EPA_OldField','EPA_Philadelphia','EPA_Pittsburgh','EPA_Queens','EPA_WashingtonDC','EPA_Westport','WBB','HoustonAldine','LibertySamHoustonLibrary','UHMoodyTower']
        #smaller subset for troubleshooting
        #locations = ['Bayonne','Bristol']
        #pods = ['EPA_Bayonne','EPA_Bristol']
        #removed 'LibertyTX' / 'LibertySamHoustonLibrary' for no bag
        
        #create a directory path for us to pull from / save to
        path = 'C:\\Users\\okorn\\Documents\\Pandora Surface\\All O3 Data Combined'
        
    elif pollutants[n] == 'SO2':
        #get the relevant location data for each
        locations = ['NewHaven','WashingtonDC','Londonderry','Bayonne','Bronx','Queens','Philadelphia','Pittsburgh','EastProvidence','Hawthorne','SWDetroit']
        pods = ['EPA_NewHaven','EPA_WashingtonDC','EPA_Londonderry','EPA_Bayonne','EPA_Bronx','EPA_Queens','EPA_Philadelphia','EPA_Pittsburgh','EPA_EastProvidence','EPA_Hawthorne','EPA_Detroit']
        
        #create a directory path for us to pull from / save to
        path = 'C:\\Users\\okorn\\Documents\\Pandora Surface\\All SO2 Data Combined'
        
    #get the model subfolder
    if bagging == 'no':    
        m = 'Outputs_{}_rf_indiv_grid'.format(pollutants[n])
    elif bagging == 'yes':
        m = 'Outputs_{}_rf_indiv_bag_grid'.format(pollutants[n])
    #combine
    mpath = path + '\\' + m
    
    #----------------------------------------------------------
    #make a function that loads our data and fits the rf algorithm
    def load_n_apply(n, pollutants, bagging, location, locations, pod, path):

        #load the data for the current location
        #---------------------------------------
        #load the Pandora (ML input) data
        filename = "{}_column_extra_{}.csv".format(location,pollutants[n])
        #combine the file and the path
        filepath = os.path.join(path, filename)
        #different load structure for FRAPPE data
        if location == 'BAO' or location == 'NREL' or location == 'Platteville':
            ann_inputs = pd.read_csv(filepath,index_col=0)
        else:
            ann_inputs = pd.read_csv(filepath,index_col=1)
        #Convert the index to a DatetimeIndex and set the nanosecond values to zero
        ann_inputs.index = pd.to_datetime(ann_inputs.index)
        #remove any empty/placeholder times
        if pollutants[n] == 'O3' and location != 'BAO' and location != 'NREL' and location != 'Platteville': 
            ann_inputs = ann_inputs[ann_inputs['Atmos Variability'] != 999]
            ann_inputs = ann_inputs[ann_inputs['O3'] != -9e99]
        
        #get rid of unnecessary FRAPPE columns
        if location == 'BAO' or location == 'NREL' or location == 'Platteville':
            ann_inputs = ann_inputs.rename(columns={'Temp-Eff': 'TEff','AMF':'O3 AMF'})
            #for FRAPPE pods, also need to add dummy columns
            ann_inputs['Atmos Variability'] = 3
            ann_inputs['O3 AMF'] = 2
            #Define the desired order of columns
            desired_order = ['SZA', 'pressure', 'O3', 'TEff', 'O3 AMF', 'Atmos Variability']
            #re-order the columns
            ann_inputs = ann_inputs[desired_order]
            #remove any nans from retime
            ann_inputs = ann_inputs.dropna()
        else: #others don't need this
            #Filter so that the lowest quality data is NOT included
            ann_inputs = ann_inputs.loc[ann_inputs['quality_flag'] != 12]
            if pollutants[n] == 'O3':
                ann_inputs = ann_inputs[['SZA','pressure','{}'.format(pollutants[n]),'TEff','{} AMF'.format(pollutants[n]),'Atmos Variability']]
            else:
                ann_inputs = ann_inputs[['SZA','pressure','{}'.format(pollutants[n]),'TEff','AMF','Atmos Variability']]
            #data cleaning - remove negatives from the pollutant column
        ann_inputs.loc[ann_inputs['{}'.format(pollutants[n])] < 0, '{}'.format(pollutants[n])] = np.nan

        #--------------------------------------------
        #now load the matching pod data
        filename = "{}_{}.csv".format(pod,pollutants[n])
        #combine the file and the path
        filepath = os.path.join(path, filename)
        if 'WBB' in pod: #wbb, slc
            ground = pd.read_csv(filepath,skiprows=10,index_col=1)
            #delete the first row - holds unit info
            ground = ground.drop(ground.index[0])
        else:
            #all others take the same format
            ground = pd.read_csv(filepath,index_col=0)  
        #Convert Index to DatetimeIndex
        if 'YPOD' in pod:
            ground.index = pd.to_datetime(ground.index, format="%d-%b-%Y %H:%M:%S")
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
            #if FRAPPE, need to change the column names
            if location == 'BAO' or location == 'NREL' or location == 'Platteville':
                ground.rename(columns={'O3':'Y_hatfield'}, inplace=True)
        
        #if it's a non-pod, need to clean the data more
        elif 'topaz' in pod: #topaz, noaa
            #need different format for non-pods
            ground.index = pd.to_datetime(ground.index, format="%Y-%m-%d %H:%M:%S")
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
            #Drop all columns except the specified one
            columns_to_drop = [col for col in ground.columns if col != 'O3_1m,ppbv']
            ground.drop(columns=columns_to_drop, inplace=True)
            #rename the main column
            ground.rename(columns={'O3_1m,ppbv':'Y_hatfield'}, inplace=True)
            #remove rows containing -999
            ground = ground[ground['Y_hatfield'] != -999]
        
        elif 'WBB' in pod: #wbb, slc
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
            #Drop all columns except the specified one
            ground = ground[['ozone_concentration_set_1']]#rename the main column
            #rename the ozone column
            ground.rename(columns={'ozone_concentration_set_1':'Y_hatfield'}, inplace=True)
            #remove rows containing -999
            ground = ground[ground['Y_hatfield'] != -999]
            #make sure our o3 data is numbers, not strings
            ground['Y_hatfield'] = ground['Y_hatfield'].astype(float)

        elif 'TX' in location: #texas data
            #this one has TX in locations - all others have signifier in 'pods'
            ground.rename(columns={'O3':'Y_hatfield'}, inplace=True)
            ground.index = pd.to_datetime(ground.index, format="%Y-%m-%d %H:%M:%S")
            #Convert the modified index to a DatetimeIndex and set the nanosecond values to zero
            ground.index = pd.to_datetime(ground.index.values.astype('datetime64[s]'), errors='coerce')
        
        elif 'EPA' in pod: #EPA data
            ground.rename(columns={'{}'.format(pollutants[n]):'Y_hatfield'}, inplace=True)
            ground.index = pd.to_datetime(ground.index)
            #convert from ppm to ppb
            ground['Y_hatfield'] = ground['Y_hatfield'] * 1000
            
        elif 'ground' in pod: #FRAPPE ref data
            ground.index = pd.to_datetime(ground.index)
            #remove whitespace from column headings
            ground.columns = ground.columns.str.strip()
            #remove rows containing -999
            ground = ground[ground['{}'.format(pollutants[n])] != -999]
            
        #-------------------------------------
        #remove any nans before retime
        ground = ground.dropna()
        #resample to hourly
        ground = ground.resample('H').mean()
        ann_inputs = ann_inputs.resample('H').mean()
            
        #combine our datasets - both already in local time
        x=pd.merge(ann_inputs,ground,left_index=True,right_index=True)
        #remove NaNs
        x = x.dropna()
            
        #Remove whitespace from column labels
        x.columns = x.columns.str.strip()
        #-------------------------------------

        #now for reformatting - get our 'y' data alone
        y = pd.DataFrame(x.pop('Y_hatfield'))
        
        #-------------------------------------
        #scale our dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)
        
        #-------------------------------------
        #-------------------------------------
        #blank slate for each location
        stats_list = []
        
        #now load the model in using a loop
        for k in locations:
            modelName = '{}_rfmodel_{}.joblib'.format(k,pollutants[n])
            #combine path and filename
            modelPath = os.path.join(mpath, modelName)
            #load the model
            model = load(modelPath)
        
            #apply the model
            y_hat = model.predict(X_scaled)
        
            #generate statistics
            r2 = r2_score(y['Y_hatfield'], y_hat)
            rmse = np.sqrt(mean_squared_error(y['Y_hatfield'], y_hat))
            mbe = np.mean(y_hat - y['Y_hatfield'])
            #store our results in a dictionary
            stats = {'AppLoc': '{}'.format(k), 'R2': r2, 'RMSE': rmse, 'MBE': mbe}
            #append
            stats_list.append(stats)
        
        #end function definition
        return stats_list
    #---------------------------------------------------
    #Function to process each pair of location and pod
    def process_location_pod(location_pod):
         location, pod = location_pod
         stats_list = load_n_apply(n, pollutants, bagging, location, locations, pod, path)
         return location, stats_list

    #Add these lines for multiprocessing on Windows
    if __name__ == '__main__':
         multiprocessing.freeze_support()

         #Pair locations and pods
         location_pod_pairs = zip(locations, pods)

         #Run parallel processing
         with concurrent.futures.ProcessPoolExecutor() as executor:
             results = list(executor.map(process_location_pod, location_pod_pairs))
#---------------------------------------------------
         # Save stats per location
         for location, stats_list in results:
             try:
                # Convert to DataFrame
                if all(isinstance(d, dict) for d in stats_list):
                     df_combined = pd.DataFrame(stats_list)
                elif all(isinstance(d, pd.DataFrame) for d in stats_list):
                    df_combined = pd.concat(stats_list, ignore_index=True)
                else:
                    print(f"Skipping {location} — unexpected entry types.")
                    continue

                #Save CSV
                save_path = os.path.join(mpath, f"{location}_stats_app_{pollutants[n]}.csv")
                df_combined.to_csv(save_path, index=False)
                print(f"Saved stats for {location} to {save_path}")

             except Exception as e:
                print(f"Error saving stats for {location}: {e}")
