



####################################################################################################################
####################################### INITAL SETUP (LIBRARIES AND IMPORTS) #######################################


### Import of needed libraries
import pandas as pd
import numpy as np


### Loading the .csv files containing data
ath_events_ds = pd.read_csv (r'Datasets\athlete_events.csv')
#print (ath_events_ds)

noc_regions_ds = pd.read_csv(r'Datasets\noc_regions.csv')
#print (noc_regions_ds)




####################################################################################################################
############################################### DATASETS EXPLORATION ###############################################


### Getting technical information 
"""
print(ath_events_ds.info())
print(noc_regions_ds.info())
"""


### Printing the first and last 20 rows of the dataset to understand content
"""
print(ath_events_ds.head(20))
print(ath_events_ds.tail(20))


print(noc_regions_ds.head(20))
print(noc_regions_ds.tail(20))
"""


### Counting the null values
"""
print(ath_events_ds.isnull().sum())
print(noc_regions_ds.isnull().sum())
"""




####################################################################################################################
################################################ DATASETS WRANGLING ################################################