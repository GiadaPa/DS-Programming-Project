



####################################################################################################################
####################################### INITAL SETUP (LIBRARIES AND IMPORTS) #######################################


### Import of needed libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


### Loading the .csv files containing data
ath_events_ds = pd.read_csv (r'Datasets\athlete_events.csv')
#print (ath_events_ds)

noc_regions_ds = pd.read_csv(r'Datasets\noc_regions.csv')
#print (noc_regions_ds)

gdp_ds = pd.read_csv(r'Datasets\gdp.csv')
#print(gdp_ds)

population_ds = pd.read_csv(r'Datasets\population.csv')
#print(population_ds)




####################################################################################################################
################################################# DATA EXPLORATION #################################################


### Getting technical information 
"""
print(ath_events_ds.info())
print(noc_regions_ds.info())
print(gdp_ds.info())
print(population_ds.info())
"""


### Printing the first and last 20 rows of the dataset to understand content
"""
print(ath_events_ds.head(20))
print(ath_events_ds.tail(20))

print(noc_regions_ds.head(20))
print(noc_regions_ds.tail(20))

print(gdp_ds.head(20))
print(gdp_ds.tail(20))

print(population_ds.head(20))
print(population_ds.tail(20))
"""


### Counting the null values
"""
print(ath_events_ds.isnull().sum())
print(noc_regions_ds.isnull().sum())
print(gdp_ds.isnull().sum())
print(population_ds.isnull().sum())
"""


### Basic statistics
"""
# AGE INFO
print(ath_events_ds['Age'].describe())

# GENDER INFO
print(ath_events_ds['Sex'].describe())

# WEIGHT
print(ath_events_ds['Weight'].describe())

# HEIGHT
print(ath_events_ds['Height'].describe())

# INFO BY GENDER
print(ath_events_ds[["Sex", "Age", "Weight", "Height"]].groupby("Sex").mean())
"""




####################################################################################################################
################################################## DATA WRANGLING ##################################################



################################################## ATHLETE_EVENTS ##################################################
### Why does the medal column have NaN values?

# Since not every athlete participating at one competition wins a medal, many rows were left empty.
# It is more readable to replace the NaN of the medal column with some explicvative text like 'NoMedal'
ath_events_ds['Medal'] = ath_events_ds['Medal'].replace(np.nan, 'NoMedal')
#print(ath_events_ds.isnull().sum())



################################################## NOC_REGIONS ##################################################
### Are all NOC (National Olympic Comittees) related to a unique team?

# Looking at the NOC dataset there's the notes column which seems pretty useless
noc_regions_ds.drop('notes', axis=1, inplace = True)

# Accessing the NOC and Team columns from 0 to the end of the dataframe
# [271115 rows x 2 columns]
check_NOC = ath_events_ds.loc[:,['NOC','Team']]

# Remove duplicate NOC and Team rows [1231 rows x 2 columns]
check_NOC_unique = check_NOC.drop_duplicates()

# Counting how many Olympic National Comittees are related to a Team
#print(check_NOC_unique['Team'].value_counts())

# Counting how many Teams are related to an Olympic National Comittee
#print(check_NOC_unique['NOC'].value_counts())

# It is impossible that a National Olympic Commitee is related to more than one team so we have to clean up the data
# So we merge the ath_events_ds with the noc_regions_ds using NOC column as primary key using a left join, so that
# if there is a NOC in the ath_events_ds that is missing in the nor_regions_ds is preserved
ath_events_ds_merge = ath_events_ds.merge(noc_regions_ds, left_on = 'NOC', right_on = 'NOC', how = 'left')

# Let's check if there is some NOC in the ath_events_ds that did not match with the NOC in the noc_regions_ds
# there are 370 empty regions
ath_events_ds_merge_is_null = ath_events_ds_merge['region'].isnull()
#print(ath_events_ds_merge_is_null.value_counts())

# Let's see which are these null NOC
#print(ath_events_ds_merge.loc[ath_events_ds_merge_is_null, ['NOC', 'Team']].drop_duplicates())

# I will enter this data manually to have a coherent 1 to 1 mapping
ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='SGP', 'Singapore', ath_events_ds_merge['region'])
ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='ROT', 'Refugee Olympic Athletes', ath_events_ds_merge['region'])
ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='UNK', 'Unknown', ath_events_ds_merge['region'])
ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='TUV', 'Tuvalu', ath_events_ds_merge['region'])

ath_events_ds_merge.drop('Team', axis = 1, inplace = True)
ath_events_ds_merge.rename(columns = {'region': 'Team'}, inplace = True)

# Checking the result
"""
print(ath_events_ds_merge['Team'].isnull().value_counts())
print(ath_events_ds_merge.head(10))
print(ath_events_ds_merge.loc[:, ['NOC', 'Team']].drop_duplicates()['NOC'].value_counts())
"""



################################################## POPULATION ##################################################
### The population dataset contains some columns that are not of interest for analysing the data 
### and for meeting my intuition so I will just remove them
population_ds.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)

# The data look awful and it's kind of useless to have each year corresponding to a column
# So I will unpivoting all the years columns to become rows with indexes Country and Code and values Year and Population
# the value_vars parameter is not set because we need all columns representing an year and in this way it unpivots them all
population_ds = pd.melt(population_ds, id_vars = ['Country', 'Country Code'], var_name = 'Year', value_name = 'Population')
#print(population_ds.info())
#print(population_ds.head())

# ERROR -> ValueError: You are trying to merge on int64 and object columns.
population_ds['Year'] = pd.to_numeric(population_ds['Year'], errors='coerce')

# Join the population_ds with the athlete_events_ds on the Team and Country key columns representing the region where the athlete comes from
ath_events_ds_merge_country = ath_events_ds_merge.merge(population_ds[['Country', 'Country Code']].drop_duplicates(), left_on = 'Team', right_on = 'Country', how = 'left')
#print(ath_events_ds_merge_country.head())

# Drop Country column since it is the same as Team
ath_events_ds_merge_country.drop('Country', axis = 1, inplace = True)
#print(ath_events_ds_merge_country.info())
#print(ath_events_ds_merge_country['Year'].head())

# Now join again on year to add the population number to the athlete_events_ds.
# This is done in a separate join, otherwise using the Team columns as key would have created for every team 60 years of population information
ath_events_ds_merge_country_pop = ath_events_ds_merge_country.merge(population_ds, left_on = ['Country Code', 'Year'], right_on = ['Country Code', 'Year'], how = 'left')
ath_events_ds_merge_country_pop.drop('Country', axis = 1, inplace = True)
#print(ath_events_ds_merge_country_pop.head())



################################################## GDP ##################################################
### Same steps performed for the population_ds, they have an identical structure
gdp_ds.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)
gdp_ds = pd.melt(gdp_ds, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'GDP')
gdp_ds['Year'] = pd.to_numeric(gdp_ds['Year'])

olympic_history_ds = ath_events_ds_merge_country_pop.merge(gdp_ds,
                                            left_on = ['Country Code', 'Year'],
                                            right_on= ['Country Code', 'Year'],
                                            how = 'left')
#print(olympic_history_ds.head())

olympic_history_ds.drop('Country Name', axis = 1, inplace = True)
#print(olympic_history_ds.head())


# Country code, population and gdp have a lot of null values.
#print(olympic_history_ds.isnull().sum())

# Checking the year we can see that the ath_events_ds has records starting from 1896, on the other hand info about population and gdp starts from 1960
def checkYearASC(ds):
    ds.sort_values(by=['Year'])
    return ds.head(1)

def checkYearDESC(ds):
    ds.sort_values(by=['Year'])
    return ds.tail(1)

"""
print(checkYearASC(population_ds))
print(checkYearDESC(population_ds))
print(checkYearASC(gdp_ds))
print(checkYearDESC(gdp_ds))
print(checkYearASC(ath_events_ds))
print(checkYearDESC(ath_events_ds))
"""

# So lets filter out 
olympic_history_ds = olympic_history_ds.loc[(olympic_history_ds['Year'] > 1960) & (olympic_history_ds['Year'] < 2017), :]
#print(olympic_history_ds.isnull().sum())
#

##
# Finally we have a complete ds with all info needed to perform some interesting analysis and prediction!
##




#################################################################################################################################
################################################ DATA ANALYSIS AND VISUALISATION ################################################


################################################## AGE DISTRIBUTION ##################################################
#removing NaN values from the ds
olympic_history_ds = olympic_history_ds[np.isfinite(olympic_history_ds['Age'])]
"""
# AGE DISTRIBUTION
plt.figure(figsize=(5, 10))
sns.countplot(olympic_history_ds['Age'])
plt.title('Age distribution', fontsize = 18)

# AGE DISTRIBUTION MALE AND FEMALE BY YEAR
fig, ax = plt.subplots(figsize=(5,10))
a = sns.boxplot(x='Year', y='Age', hue='Sex', palette={'M':'blue', 'F':'pink'}, data=olympic_history_ds, ax=ax)      
ax.set_xlabel('Year', size=14)
ax.set_ylabel('Age', size=14)
ax.set_title('Age distribution by year', fontsize=18)
"""

# Who is the oldest athlete?
pd.set_option('max_columns', 18)
#print(olympic_history_ds[olympic_history_ds['Age'] == olympic_history_ds['Age'].max()].drop_duplicates(subset=['Name']))


# And who's the youngest?
pd.set_option('max_columns', 18)
#print(olympic_history_ds[olympic_history_ds['Age'] == olympic_history_ds['Age'].min()].drop_duplicates(subset=['Name']))


# The oldest athlete is an athlete that practices equestrianism.
# Lets see which sport do elder athlete practice
olympic_history_ds_age = olympic_history_ds['Sport'][olympic_history_ds['Age'] > 60]


plt.figure(figsize=(5, 10))
sns.countplot(x=olympic_history_ds_age, data=olympic_history_ds, order = olympic_history_ds_age.value_counts().index)
plt.title('Athletes Over 60')
plt.show()
