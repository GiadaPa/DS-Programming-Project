
####################################################################################################################
####################################### INITAL SETUP (LIBRARIES AND IMPORTS) #######################################

### Import of needed libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
import streamlit as st

### Loading the .csv files containing data
ath_events_ds = pd.read_csv (r'Datasets\athlete_events.csv')
#print (ath_events_ds)

noc_regions_ds = pd.read_csv(r'Datasets\noc_regions.csv')
#print (noc_regions_ds)

gdp_ds = pd.read_csv(r'Datasets\gdp.csv')
#print(gdp_ds)

population_ds = pd.read_csv(r'Datasets\population.csv')
#print(population_ds)


#--------------------------------------------------------------------------------------------------------------------------------
# STREAMLIT ---------------------------------------------------------------------------------------------------------------------
img = Image.open("olympics.jpg")
img_h = Image.open("olympics-header.jpg")

st.set_page_config(page_title="Olympic Games EDA - Giada Palma", page_icon=img)
st.markdown(""" 
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style> """, 
    unsafe_allow_html=True
)
st.image(img_h)
st.title("THE OLYMPIC GAMES")


rad_navigation = st.sidebar.radio("Navigation", ["INITIAL SETUP","DATA EXPLORATION", "DATA WRANGLING", "DATA ANALYSIS", "PREDICTION"])

if rad_navigation == "INITIAL SETUP":
    st.header("INITIAL SETUP")
    st.text("As first step, for the implementation of this project, I have imported the needed libraries and the .csv files containing the data")
    st.code('''import pandas as pd    
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt''', language="python") 
    st.write("The dataset of interest are the following:  \n* Olympics History and NOC [link](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)  \n* GDP [link](https://www.kaggle.com/chadalee/country-wise-gdp-data)  \n* Population [link](https://www.kaggle.com/centurion1986/countries-population)")   
    
    st.code('''
ath_events_ds = pd.read_csv (r'Datasets\\athlete_events.csv')
noc_regions_ds = pd.read_csv(r'Datasets\\noc_regions.csv')
gdp_ds = pd.read_csv(r'Datasets\gdp.csv')
population_ds = pd.read_csv(r'Datasets\population.csv')
    ''')
#--------------------------------------------------------------------------------------------------------------------------------


####################################################################################################################
################################################# DATA EXPLORATION #################################################
### Basic statistics
'''
# AGE INFO
print(ath_events_ds['Age'].describe())

# GENDER INFO
print(ath_events_ds['Sex'].describe())

# WEIGHT
print(ath_events_ds['Weight'].describe())

# HEIGHT
print(ath_events_ds['Height'].describe())
'''

### Counting the null values
def show_nulls(df):
    nulls = df.isnull().sum()
    return nulls

ath_mean = ath_events_ds[['Sex', 'Age', 'Weight', 'Height']].groupby('Sex').mean()

correlation = ath_events_ds.corr()
fig, ax = plt.subplots(figsize=(10,6))
ax = sns.heatmap(data=correlation, cmap="YlGnBu", annot=True, ax=ax)



#--------------------------------------------------------------------------------------------------------------------------------
# STREAMLIT ---------------------------------------------------------------------------------------------------------------------
if rad_navigation == "DATA EXPLORATION":
    st.header("DATA EXPLORATION")
    st.text("As second step, I have explored the datasets previously imported to find some correaltion, some unexpected info and NaN values.")
    option = st.selectbox(
        'Select wich datset to view',
        ('Athlete', 'NOC', 'GDP', 'Population'))
    st.write('You are visualising data from the dataset:', option)
    if option == 'Athlete':
        st.write(ath_events_ds)        
        st.subheader("NaN VALUES")
        st.text("Let's see if there are some NaN entries")
        st.write(show_nulls(ath_events_ds))
        st.text("There are strange NaN values in the medal column. I suppose this is due to the fact that not any athlete wins a medal.")   
        st.subheader("MEAN VALUES")
        st.text("  \nLet's see the average values for height, weight and age of athletes grouped by gender")
        st.write(ath_mean)
        st.subheader("CORRELATION")
        st.text("Let's see correlation in the data")
        st.pyplot(fig)
        st.text("There is a strong correlation between height and weight (equals to 0.8) and a discrete correlation between weight and age (equals to 0.21).")
    if option == 'NOC':
        st.write(noc_regions_ds)
        st.subheader("NaN VALUES")
        st.write(show_nulls(noc_regions_ds))
    if option == 'GDP':
        st.write(gdp_ds)
        st.subheader("NaN VALUES")
        st.write(show_nulls(gdp_ds))
    if option == 'Population':
        st.write(population_ds)
        st.subheader("NaN VALUES")
        st.write(show_nulls(population_ds))
#--------------------------------------------------------------------------------------------------------------------------------

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

olympic_history_ds = ath_events_ds_merge_country_pop.merge(gdp_ds, left_on = ['Country Code', 'Year'], right_on= ['Country Code', 'Year'], how = 'left')
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
olympic_history_ds = olympic_history_ds.loc[(olympic_history_ds['Year'] > 1960) & (olympic_history_ds['Year'] < 2016), :]
#print(olympic_history_ds.isnull().sum())
#

##
# Finally we have a complete ds with all info needed to perform some interesting analysis and prediction!
##


#--------------------------------------------------------------------------------------------------------------------------------
# STREAMLIT ---------------------------------------------------------------------------------------------------------------------
if rad_navigation == "DATA WRANGLING":
    st.header("DATA WRANGLING")
    st.subheader("As third step, I have performed some data cleansing on all the datasets and then combined them all together in a single one.")
    st.subheader("WRANGLING ON ATHLETE_EVENTS_DS")
    st.text("I have decided to replace the NaN values of the medal column with a text saying \"NoMedal\"")
    st.code('''ath_events_ds['Medal'] = ath_events_ds['Medal'].replace(np.nan, 'NoMedal')''')
    st.subheader("WRANGLING ON NOC_DS")
    st.text("Firstly, I have removed the notes column, which contains many NaN values and seems useless.")
    st.code('''noc_regions_ds.drop('notes', axis=1, inplace = True)''')    
    st.text("  \nSecondly, I want to check if there is a unique Team for each NOC.  \nTo do this I have removed duplicate values, and checked how many teams are related to a NOC.")
    st.code('''check_NOC_unique = check_NOC.drop_duplicates()
check_NOC_unique['Team'].value_counts()''')
    st.text("  \nThirdly, I have merged the ath_event_ds with the noc_regions to create unique relations between team and NOC.")
    st.code('''ath_events_ds_merge = ath_events_ds.merge(noc_regions_ds, left_on = 'NOC', right_on = 'NOC', how = 'left')''')
    st.text("  \nLet's check if there are still some NOC that do not match any Team in the newly reated ds.")
    st.write(ath_events_ds_merge.loc[ath_events_ds_merge_is_null, ['NOC', 'Team']].drop_duplicates())
    st.text("  \nI will add them manually in the ds.")
    st.code('''ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='SGP', 'Singapore', ath_events_ds_merge['region'])
ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='ROT', 'Refugee Olympic Athletes', ath_events_ds_merge['region'])
ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='UNK', 'Unknown', ath_events_ds_merge['region'])
ath_events_ds_merge['region'] = np.where(ath_events_ds_merge['NOC']=='TUV', 'Tuvalu', ath_events_ds_merge['region'])

ath_events_ds_merge.drop('Team', axis = 1, inplace = True)
ath_events_ds_merge.rename(columns = {'region': 'Team'}, inplace = True)''')
    st.text("  \nCheck the ds after the first two \"wrangling\" steps.")
    st.write(ath_events_ds_merge)
    st.subheader("WRANGLING ON POPULATION_DS")
    st.text("Firstly, I have removed the useless Indicator Name and Code columns. Then I have performed an unpivoting operation to make the years column become row values.")
    st.code('''population_ds.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)
population_ds = pd.melt(population_ds, id_vars = ['Country', 'Country Code'], var_name = 'Year', value_name = 'Population')''')
    st.text("  \nSecondly, I have merged the previously obtained ds with this one using the Country column as key. And then merged the result ds again with itself on Country and Year column ")
    st.code('''ath_events_ds_merge_country = ath_events_ds_merge.merge(population_ds[['Country', 'Country Code']].drop_duplicates(), left_on = 'Team', right_on = 'Country', how = 'left')''')
    st.text("  \nLet's check the result.")
    st.write(ath_events_ds_merge_country_pop)
    st.subheader("WRANGLING ON GDP_DS")
    st.text("Firstly, I have performed the same steps I did with the population_ds.")
    st.code('''gdp_ds.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)
gdp_ds = pd.melt(gdp_ds, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'GDP')
gdp_ds['Year'] = pd.to_numeric(gdp_ds['Year'])

olympic_history_ds = ath_events_ds_merge_country_pop.merge(gdp_ds, left_on = ['Country Code', 'Year'], right_on= ['Country Code', 'Year'], how = 'left')''')
    st.text("  \nSecondly, by checking the ds, there are a lot of null values in the year, gdp and population column. By performing some exploration it resulted that the year span of the ds merged together do not match.")
    st.text("So I have decided to reduce the year span from 1960 to 2016")
    st.code('''olympic_history_ds = olympic_history_ds.loc[(olympic_history_ds['Year'] > 1960) & (olympic_history_ds['Year'] < 2016), :]''')
    st.subheader("FINALLY")
    st.text("Let's see the dataset that I have used to performed the analysis step.")
    st.write(olympic_history_ds)
#--------------------------------------------------------------------------------------------------------------------------------



#################################################################################################################################
################################################ DATA ANALYSIS AND VISUALISATION ################################################



################################################## AGE DISTRIBUTION ##################################################
# Removing NaN values from the ds
olympic_history_ds = olympic_history_ds[np.isfinite(olympic_history_ds['Age'])]

'''
# AGE DISTRIBUTION
age_dis, ax = plt.subplots(figsize=(10, 5))
ax = sns.countplot(x='Age', data = olympic_history_ds, ax=ax)
ax.set_xlabel('Age', size=12)
ax.set_ylabel('Count', size=12)
ax.set_title('Age distribution', fontsize=16)
'''

# AGE DISTRIBUTION MALE AND FEMALE BY YEAR
fig_age_m_f, ax = plt.subplots(figsize=(10,5))
ax = sns.boxplot(x='Year', y='Age', hue='Sex', palette={'M':'blue', 'F':'pink'}, data = olympic_history_ds, ax=ax)      
ax.set_xlabel('Year', size=12)
ax.set_ylabel('Age', size=12)
ax.set_title('Age distribution by year and gender', fontsize=16)


# Who is the oldest athlete?
oldest = olympic_history_ds[olympic_history_ds['Age'] == olympic_history_ds['Age'].max()].drop_duplicates(subset=['Name'])[['Name','Sex','Age','Sport','Team','NOC','Games']]
# And who's the youngest?
youngest = olympic_history_ds[olympic_history_ds['Age'] == olympic_history_ds['Age'].min()].drop_duplicates(subset=['Name'])[['Name','Sex','Age','Sport','Team','NOC','Games']]

# The oldest athlete is an athlete that practices equestrianism.
# Lets see which sport do elder athlete practice
olympic_history_ds_age = olympic_history_ds['Sport'][olympic_history_ds['Age'] > 60]

fig_elder, ax = plt.subplots(figsize=(10, 5))
ax = sns.countplot(x=olympic_history_ds_age, data=olympic_history_ds, order=olympic_history_ds_age.value_counts().index)
ax.set_xlabel('Sport', size=12)
ax.set_ylabel('Count', size=12)
ax.set_title('Sports played by athletes over 60', fontsize=16)


# We want to analyse the mean age of athletes who won a medal
olympic_history_ds_medalists_age = olympic_history_ds.pivot_table(olympic_history_ds, index=['Year','Medal'], aggfunc=np.mean).reset_index()[['Year','Medal','Age']]
olympic_history_ds_medalists_age = olympic_history_ds_medalists_age.pivot("Medal", "Year", "Age")
olympic_history_ds_medalists_age = olympic_history_ds_medalists_age.reindex(["Gold","Silver","Bronze"])

fig_avg_age_medallist, ax = plt.subplots(figsize=(15, 6))
ax = sns.heatmap(olympic_history_ds_medalists_age, annot=True, linewidths=0.05, ax=ax, cmap="YlGnBu")
ax.set_xlabel('Year', size=12)
ax.set_ylabel('Medal', size=12)
ax.set_title('Mean age of medallists', fontsize=16)



################################################## GENDER DISTRIBUTION ##################################################
# Distribution of athletes by gender
olympic_history_ds_gender = olympic_history_ds.loc[:,['Year', 'ID', 'Sex']].drop_duplicates().groupby(['Year','Sex']).size().reset_index()
olympic_history_ds_gender.columns = ['Year','Sex','Count']

palette = {'M': 'tab:blue','F': 'tab:pink'}

fig_gender_dist, ax = plt.subplots(figsize=(10,5))
ax = sns.barplot(x='Year', y='Count', data=olympic_history_ds_gender, hue='Sex', palette=palette, ax=ax)
ax.set_xlabel('Year', size=12)
ax.set_ylabel('Count', size=12)   
ax.set_title('Number of female & male athletes by years', fontsize=16) 


########################## WOMEN ANALYSIS ##########################
# We create a subset of the ds with only women
olympic_history_ds_women = olympic_history_ds[olympic_history_ds.Sex == 'F'].drop_duplicates(subset=['Name'])
#print(olympic_history_ds_women.head())

# We want to see the 20 most practiced sports by women
olympic_history_ds_women_sport = olympic_history_ds_women['Sport']

fig_women_sports, ax = plt.subplots(figsize=(10,5))
ax = sns.countplot(x=olympic_history_ds_women_sport, data=olympic_history_ds_women_sport, order=olympic_history_ds_women_sport.value_counts()[:20].index)
plt.xticks(rotation=65)
ax.set_xlabel('Sport', size=12)
ax.set_ylabel('Count', size=12)   
ax.set_title('Most practised sports by women', fontsize=16) 



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
#Create a pivoted ds to show the number of medals won by nation by year in a linechart
medagliere_by_year_piv = pd.pivot_table(medagliere, index = 'Year', columns = 'Team', values = 'Medal_i', aggfunc = 'sum')[top6_nations]
#print(medagliere_by_year_piv)

#medagliere_by_year_piv_df = medagliere_by_year_piv.reset_index()
#print(medagliere_by_year_piv_df)

fig_medagliere, ax = plt.subplots(figsize=(10,5))
ax = sns.lineplot(data=medagliere_by_year_piv)
plt.xticks(rotation=65)
ax.set_xlabel('Year', size=12)
ax.set_ylabel('Nation', size=12)   
ax.set_title('Medals won by nations over the years', fontsize=16) 


# Create a mask to match 6 best nations
top6_nations_mask = olympic_history_ds_nation['Team'].map(lambda x: x in top6_nations)

# Create a pivoted ds to calculate sum of gold, silver and bronze medals for each nation
medagliere_medals = pd.pivot_table(olympic_history_ds_nation[top6_nations_mask], index = ['Team'], columns = 'Medal', values = 'Medal_i', aggfunc = 'sum', fill_value = 0).drop('NoMedal', axis = 1)
#print(medagliere_medals)
# Order medals to be shown on the barchart
medagliere_medals = medagliere_medals.loc[:, ['Gold', 'Silver', 'Bronze']]
#print(medagliere_medals)

# GRAPH: Gold, silver, bronze distribution by nations line 511



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
correlation_gdp_medals = medagliere_gdp.loc[medal_i_mask, ['GDP', 'Medal_i']].corr()['Medal_i'][0]
# This is a quite high correlation
#print(correlation)

fig_gdp_distr, ax = plt.subplots(figsize=(10,5))
ax = sns.scatterplot(data=medagliere_gdp, x=medagliere_gdp.loc[medal_i_mask, 'GDP'], y=medagliere_gdp.loc[medal_i_mask, 'Medal_i'], ax=ax)
ax.set_xlabel('GDP', size=12)
ax.set_ylabel('Medals', size=12)  
ax.set_title('Medals distribution based on GDP', fontsize=16) 



# Gold, silver, bronze distribution by nations
fig_medal_distr, ax = plt.subplots()
ax = medagliere_medals.plot(kind = 'bar', stacked = True, figsize=(10,5), rot = 0, color=['gold', 'silver', 'brown'])
ax.set_xlabel('Nation', size=12)
ax.set_ylabel('Medals', size=12)  
ax.set_title('Gold, silver, bronze distribution by nations', fontsize=16) 

#--------------------------------------------------------------------------------------------------------------------------------
# STREAMLIT ---------------------------------------------------------------------------------------------------------------------
if rad_navigation == "DATA ANALYSIS":
    st.header("DATA ANALYSIS AND VISUALISATION")
    st.subheader("As fourth step, I have performed some data analysis and I propose some graphical visualisation to understand better the data.")

    st.subheader("AGE DISTRIBUTION")
    st.text("First of all, I have removed NaN values.")
    st.code('''olympic_history_ds = olympic_history_ds[np.isfinite(olympic_history_ds['Age'])]''')
    st.text("Let's see the age distribution of athletes plotted against year and gender.")
    #st.pyplot(age_dis)
    st.pyplot(fig_age_m_f)
    st.text("Let's see who are the oldest and youngest athletes.")
    st.text("OLDEST")
    st.write(oldest)
    st.text("YOUNGEST")
    st.write(youngest)
    st.text("It's obvious to see that an elder athlete is not a swimmer or a 100m runner.   \nBut let's discover which are the most practiced sports by elders.")
    st.code('''olympic_history_ds_elder = olympic_history_ds['Sport'][olympic_history_ds['Age'] > 60]''')
    st.pyplot(fig_elder)
    st.text('As expected elder athletes practice sports that do not require resistance or strength.')
    st.text("It is also interesting to see the average age of medallist athletes.  \nLet's discover it by creating a pivot table with Medal and Age index columns.")
    st.code('''olympic_history_ds_medallists_age = olympic_history_ds.pivot_table(olympic_history_ds, index=['Age','Medal'], aggfunc=np.mean).reset_index()[['Year','Medal','Age']]
olympic_history_ds_medallists_age = olympic_history_ds_medallists_age.pivot("Medal", "Year", "Age")
olympic_history_ds_medallists_age = olympic_history_ds_medallists_age.reindex(["Gold","Silver","Bronze"])''')
    st.pyplot(fig_avg_age_medallist)

    st.subheader("GENDER DISTRIBUTION")
    st.text("Let's see the distribution of male and female per year.")
    st.code('''olympic_history_ds_gender = olympic_history_ds.loc[:,['Year', 'ID', 'Sex']].drop_duplicates().groupby(['Year','Sex']).size().reset_index()
olympic_history_ds_gender.columns = ['Year','Sex','Count']''')
    st.pyplot(fig_gender_dist)
    st.subheader("FEMALE ATHLETES AT THE OLYMPICS")
    st.text("Firstly, I have filtered data by gender.  \nLet's see sports distribution for female athletes. ")
    st.code('''olympic_history_ds_women = olympic_history_ds[olympic_history_ds.Sex == 'F'].drop_duplicates(subset=['Name'])''')
    st.pyplot(fig_women_sports)
    st.text("INTERACTIVE SEARCH")
    input_sport = st.text_input('Insert a sport (starting with capital letter, eg. Weightlifting) to know how many women practice this sport')
    olympic_history_ds_women_wl = len(olympic_history_ds_women[olympic_history_ds_women.Sport == input_sport].drop_duplicates(subset=['Name']))
    st.write('The searched sport is: ', input_sport)
    st.write('There are: ', olympic_history_ds_women_wl, ' practicing this sport.')

    st.subheader("MEDALS DISTRIBUTION")
    st.text("In order to get correct information about medals won, I have to take into consideration team events and single events, to avoid duplicate counting.")
    st.text("Firstly, I have created an index column containing 1 for a won medal and 0 otherwise.")
    st.code('''olympic_history_ds['Medal_i'] = np.where(olympic_history_ds.loc[:,'Medal'] == 'NoMedal', 0, 1)''')
    st.text("Secondly, I have identified which are the team events of the olympics.")
    st.code('''team_medal = pd.pivot_table(olympic_history_ds, index = ['Team', 'Year', 'Event'], columns = 'Medal', values = 'Medal_i', aggfunc = 'sum', fill_value = 0).drop('NoMedal', axis = 1).reset_index()
team_medal = team_medal.loc[team_medal['Gold'] > 1, :]
team_event = team_medal['Event'].unique()''')
    st.write(team_event)
    st.text("Then, I have written some masks to create a column 'Event_cat' that identifies if the event is team or single.")
    st.code('''team_event_mask = olympic_history_ds['Event'].map(lambda x: x in team_event)
single_event_mask = [not i for i in team_event_mask]
medal_mask = olympic_history_ds['Medal_i'] == 1

olympic_history_ds['T_event'] = np.where(team_event_mask & medal_mask, 1, 0)
olympic_history_ds['S_event'] = np.where(single_event_mask & medal_mask, 1, 0)

olympic_history_ds['Event_cat'] = olympic_history_ds['S_event'] + olympic_history_ds['T_event']''')
    st.text("Then, I have grouped data by year, team, event and medal.")
    st.code('''olympic_history_ds_nation = olympic_history_ds.groupby(['Year', 'Team', 'Event', 'Medal'])[['Medal_i', 'Event_cat']].agg('sum').reset_index()
olympic_history_ds_nation['Medal_i'] = olympic_history_ds_nation['Medal_i']/olympic_history_ds_nation['Event_cat']''')
    st.text("And then, grouped again by year and team to sum the won medals.  \nFinally, I have created a pivot table to show the top 10 nations by medals won.")
    st.code('''medagliere = olympic_history_ds_nation.groupby(['Year','Team'])['Medal_i'].agg('sum').reset_index()
medagliere_piv = pd.pivot_table(medagliere, index = 'Team', columns = 'Year', values = 'Medal_i', aggfunc = 'sum', margins = True)''')
    st.text("Let's see which are the top 10 powers of the Olympic Games.")
    st.write(top10_nations)
    st.text("Let's see the top 6 nations plotted on a graph.")
    st.pyplot(fig_medagliere)
    st.text("Lastly, I want to see the distribution of gold, silver and bronze medals for the top 6 nations.  \n So, I have created a mask to map them.")
    st.code('''top6_nations_mask = olympic_history_ds_nation['Team'].map(lambda x: x in top6_nations)''')
    st.text("Then, I have crated a pivot table indexed on Team  with Medal as column.")
    st.code('''medagliere_medals = pd.pivot_table(olympic_history_ds_nation[top6_nations_mask], index = ['Team'], columns = 'Medal', values = 'Medal_i', aggfunc = 'sum', fill_value = 0).drop('NoMedal', axis = 1)''')
    st.text("Let's see the distribution on a bar chart.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig_medal_distr=plt)
    
    st.subheader("MEDALS DISTRIBUTION BASED ON GDP")
    st.text("I suppose the number of medals won by the 6 best nations is strictly related to the GDP of that nation.")
    st.text("To verify this hypothesis, let's see the correlation between GDP and the sum of all medals won.")
    st.code('''medal_i_mask = medagliere_gdp['Medal_i'] > 0
correlation = medagliere_gdp.loc[medal_i_mask, ['GDP', 'Medal_i']].corr()['Medal_i'][0]''')
    st.write(correlation_gdp_medals)
    st.text("As expected the correlation is pretty high.")
    st.text("Let's see also a scatterplot with the medals won plotted against GDP of nations.")
    st.pyplot(fig_gdp_distr)
  
#--------------------------------------------------------------------------------------------------------------------------------





#################################################################################################################################
################################################## SPORT PREDICTION BASED ON WEIGHT AND HEIGHT ##################################################





