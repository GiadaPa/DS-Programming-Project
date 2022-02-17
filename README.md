# DS_ProgrammingProject
Project for the programming course (Master's Degree in Data Science)


# GIADA PALMA - VR471280


## DATASET URL 
  \nOLYMPICS DATA https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results
  \nGDP https://www.kaggle.com/chadalee/country-wise-gdp-data
  \nWORLD POPULATION https://www.kaggle.com/centurion1986/countries-population



## BASIC DATASET EXPLORATION
    Using info(), head(), tail() isnull(), describe() and groupby() methods


## DATA WRANGLING 
I have:
  \nReplaced null values in the noc_regions_ds with meaningful information
  \nCreated unique values for the NOC (National Olympic Committee) column corresponding each to a specific region value
  \nJoined the dataset of the athlete_events with the dataset of the population
  \nJoined the dataset obtained from the above join with the dataset of the gdp to have all information in one dataset called 'olympic_history_ds'
  \nPerformed some filtering to remove some null values from gdp and population columns due to differences in the year span of the datasets


## DATA ANALYSIS 
    1. Age distribution, age distribution by year, elder athletes's practiced sports distribution
    2. Gender distribution per sport and female partecipation history
    3. Medal distribution and GDP correlation

## PREDICTION
   1. Which sport is the most suitable for an athlete based on sex, height, weight and age?

## STREAMLIT LIBRARY TO PRESENT THE PROJECT

## HOW TO RUN THE PROJECT
Clone the directory [link]https://github.com/GiadaPa/DS_ProgrammingProject.git
Inside the directory run a terminal with the python command
```python
streamlit run OlympicHistory.py
```
This will open a browser with the project running.

# IMPORTANT
Usually it takes 1 minute to load the content. Be patient and enjoy with the interactive parts! :) 