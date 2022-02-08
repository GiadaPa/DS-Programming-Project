# DS_ProgrammingProject
Project for the programming course (Master's Degree in Data Science)


# GIADA PALMA - VR471280


DATASET URL 
    OLYMPICS DATA https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results

    GDP https://www.kaggle.com/chadalee/country-wise-gdp-data

    WORLD POPULATION https://www.kaggle.com/centurion1986/countries-population



BASIC DATASET EXPLORATION
    Using info(), head(), tail() isnull(), describe() and groupby() methods


DATA WRANGLING
    Replaced null values in the noc_regions_ds with meaningful information
    Created unique values for the NOC (National Olympic Committee) column corresponding each to a specific region value
    Joined the dataset of the athlete_events with the dataset of the population
    Joined the dataset obtained from the above join with the dataset of the gdp to have all information in one dataset called 'olympic_history_ds'
    Performed some filtering to remove some null values from gdp and population columns due to differences in the year span of the datasets


DATA ANALYSIS
    1. Age distributions
    2. Gender distribution per sport and female partecipation history
    3. Partecipation frequency of Nations

PREDICTION
   1. Which sport is the most suitable for an athlete based on weight and age?
   2. Which are the top 5 favourite athletes to win a medal in weightlifting next year?
