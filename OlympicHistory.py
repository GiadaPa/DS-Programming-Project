



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
print(ath_events_ds[['Sex', "Age', "Weight', "Height']].groupby('Sex').mean())
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
sns.countplot(x=olympic_history_ds_age, data=olympic_history_ds, order=olympic_history_ds_age.value_counts().index)
plt.title('Athletes Over 60')


# We want to analyse the mean age of athletes who won a medal
olympic_history_ds_medalists_age = olympic_history_ds.pivot_table(olympic_history_ds, index=['Year','Medal'], aggfunc=np.mean).reset_index()[['Year','Medal','Age']]
olympic_history_ds_medalists_age = olympic_history_ds_medalists_age.pivot("Medal", "Year", "Age")
olympic_history_ds_medalists_age = olympic_history_ds_medalists_age.reindex(["Gold","Silver","Bronze"])

f, ax = plt.subplots(figsize=(20, 3))
sns.heatmap(olympic_history_ds_medalists_age, annot=True, linewidths=0.05, ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Medal')
ax.set_title('Mean age of medalists')
plt.show()
"""



################################################## GENDER DISTRIBUTION ##################################################

# Distribution of athletes by gender
olympic_history_ds_gender = olympic_history_ds.loc[:,['Year', 'ID', 'Sex']].drop_duplicates().groupby(['Year','Sex']).size().reset_index()
olympic_history_ds_gender.columns = ['Year','Sex','Count']
"""
plt.figure(figsize=(10,10))
sns.barplot(x='Year', y='Count', data=olympic_history_ds_gender, hue='Sex')   
plt.title('Number of Female & Male Athletes by Years') 
plt.show()
"""


########################## WOMEN ANALYSIS ##########################

# We create a subset of the ds with only women
olympic_history_ds_women = olympic_history_ds[olympic_history_ds.Sex == 'F']
#print(olympic_history_ds_women.head())

# We want to see the 20 most practiced sports by women
olympic_history_ds_women_sport = olympic_history_ds_women['Sport']
"""
plt.figure(figsize=(15, 10))
sns.countplot(x=olympic_history_ds_women_sport, data=olympic_history_ds_women_sport, order=olympic_history_ds_women_sport.value_counts()[:20].index)
plt.xticks(rotation=45)
plt.title('Most practised sport by women')
plt.show()
"""

# I am interested in knowing how many women have practiced weightlifting in the history of the olympic games
olympic_history_ds_women_wl = olympic_history_ds_women[olympic_history_ds_women.Sport == 'Weightlifting'].drop_duplicates(subset=['Name']).value_counts().to_frame()
#print(olympic_history_ds_women.drop_duplicates(subset=['Name']))
#print(olympic_history_ds_women_wl)



################################################## MEDAL DISTRIBUTION AND GDP CORRELATION ##################################################

########################## MEDAGLIERE ##########################
# Create indexing column counting medal or not
olympic_history_ds['Medal_i'] = np.where(olympic_history_ds.loc[:,'Medal'] == 'NoMedal', 0, 1)
#print(olympic_history_ds['Medal_i'])

# From the ds we know if an athlete wins a medal or not. But thinking about team events, each athlete receives a medal. 
# So the total sum would be incorrect if we think about a sum of medals won by a nation
# We have to identify team events and single events
team_medal = pd.pivot_table(olympic_history_ds, index = ['Team', 'Year', 'Event'], columns = 'Medal', values = 'Medal_i', aggfunc = 'sum', fill_value = 0).drop('NoMedal', axis = 1).reset_index()

team_medal = team_medal.loc[team_medal['Gold'] > 1, :]
team_event = team_medal['Event'].unique()
#print(team_event)

# Now we have to add a column that identifies if the event is team or single so that when we sum the medals by nation we have the correct counting.
# Create a mask for when an event is a team event, otherwise identifies it as single event
team_event_mask = olympic_history_ds['Event'].map(lambda x: x in team_event)
single_event_mask = [not i for i in team_event_mask]

# Create a mask for when an entry in medal_i is 1, i.e won a medal
medal_mask = olympic_history_ds['Medal_i'] == 1

# Identify with 1 if it is a team event or a single event
olympic_history_ds['T_event'] = np.where(team_event_mask & medal_mask, 1, 0)
olympic_history_ds['S_event'] = np.where(single_event_mask & medal_mask, 1, 0)

# Add a column that contains single or team events
olympic_history_ds['Event_cat'] = olympic_history_ds['S_event'] + olympic_history_ds['T_event']
#print(olympic_history_ds.head())

#Groupby year, team, event and medal and then count the medal won
olympic_history_ds_nation = olympic_history_ds.groupby(['Year', 'Team', 'Event', 'Medal'])[['Medal_i', 'Event_cat']].agg('sum').reset_index()
olympic_history_ds_nation['Medal_i'] = olympic_history_ds_nation['Medal_i']/olympic_history_ds_nation['Event_cat']
#pd.set_option('max_columns', 18)
#print(olympic_history_ds_nation.head())

# Groupby year and team to create a medal recap
medagliere = olympic_history_ds_nation.groupby(['Year','Team'])['Medal_i'].agg('sum').reset_index()

# Create a pivoted ds to show the top 10 nations by medal won by year
medagliere_piv = pd.pivot_table(medagliere, index = 'Team', columns = 'Year', values = 'Medal_i', aggfunc = 'sum', margins = True)
top10_nations = medagliere_piv.sort_values('All', ascending = False).head(10)
#print(top10_nations)

# Create a list of the 6 best nations by medals won
top6_nations = ['USA', 'Russia', 'Germany', 'China', 'France', 'Italy']

#Create a pivoted ds to show the number of medals won by nation by year in a linechart
medagliere_by_year_piv = pd.pivot_table(medagliere, index = 'Year', columns = 'Team', values = 'Medal_i', aggfunc = 'sum')[top6_nations]
"""
medagliere_by_year_piv.plot(linestyle = '-', figsize = (10,8), linewidth = 1)
plt.xlabel('Year')
plt.ylabel('Medals')
#plt.show()
"""
# Create a mask to match 6 best nations
top6_nations_mask = olympic_history_ds_nation['Team'].map(lambda x: x in top6_nations)

# Create a pivoted ds to calculate sum of gold, silver and bronze medals for each nation
medagliere_medals = pd.pivot_table(olympic_history_ds_nation[top6_nations_mask], index = ['Team'], columns = 'Medal', values = 'Medal_i', aggfunc = 'sum', fill_value = 0).drop('NoMedal', axis = 1)

# Order medals to be shown on the barchart
medagliere_medals = medagliere_medals.loc[:, ['Gold', 'Silver', 'Bronze']]
"""
medagliere_medals.plot(kind = 'bar', stacked = True, figsize = (8,6), rot = 0, color=['gold', 'silver', 'brown'])
plt.xlabel('Nation')
plt.ylabel('Medals')
plt.show()
"""


########################## GDP ##########################
# I suppose the number of medals won by the 6 best nations is strictly related to the GDP of that nation
# So for example a very poor nation like Zimambwe has won no medal at all
# To verify my supposition:

# I create a ds filtering on year team and gdp and removing duplicates
year_team_gdp = olympic_history_ds.loc[:, ['Year', 'Team', 'GDP']].drop_duplicates()

# I merge the medagliere ds with the newly created the line above with a left join to keep all values from the medagliere ds
medagliere_gdp = medagliere.merge(year_team_gdp, left_on = ['Year', 'Team'], right_on = ['Year', 'Team'], how = 'left')

# Create a medal mask that should be 1
medal_i_mask = medagliere_gdp['Medal_i'] > 0

# calculate the correlation of GDP and the medal won
correlation = medagliere_gdp.loc[medal_i_mask, ['GDP', 'Medal_i']].corr()['Medal_i'][0]
# This is a quite high correlation
#print(correlation)
"""
# Let's plot on the chart the GDP on the x axis and the medals on the y
plt.plot(medagliere_gdp.loc[medal_i_mask, 'GDP'],  medagliere_gdp.loc[medal_i_mask, 'Medal_i'], linestyle='none', marker = 'o', alpha = 0.4)
plt.xlabel('GDP')
plt.ylabel('Medals')
plt.show()
"""



#################################################################################################################################
################################################## SPORT PREDICTION BASED ON WEIGHT AND HEIGHT ##################################################
