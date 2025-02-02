#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install pandas matplotlib seaborn


# In[18]:


import pandas as pd #importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


data =pd.read_csv(r'C:\Users\pc\Downloads\API_SP.POP.TOTL_DS2_en_csv_v2_900.csv') #loading the dataset
data.head(), data.info()


# In[22]:


missing_data = data.isnull() #using the isnull() function to check for missing values in the dataset
missing_data.head(5)


# In[24]:


x= missing_data.columns
y = x.values
y


# In[26]:


missing_data


# In[28]:


#Filtering data for the year 2020
population_2020 = data[['Country Name','2020']].dropna()
population_2020 = population_2020.sort_values('2020',ascending=False).head(20) #Top 20 countries by population


# In[30]:


# PLotting a bar chart for the top 20 most populous countries in 2020
plt.figure(figsize=(12,8))
plt.barh(population_2020['Country Name'], population_2020['2020'], color='red')
plt.xlabel('population(2020)',fontsize=12)
plt.ylabel('Country',fontsize=12)
plt.title('Top 20 Most Populous Countries in 2020', fontsize=14)
plt.gca(). invert_yaxis() #Invert y-axis for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()


# In[40]:


population_2020 =data[['Country Name', '2020']].dropna()
population_2020 = population_2020.sort_values('2020', ascending=True).head(20)
#Plottting a bar chart for the Top 20 least populous countries in 2020
plt.figure(figsize=(12, 8))
plt.barh(population_2020['Country Name'], population_2020['2020'], color='green')
plt.xlabel('population (2020)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.title('Top 20 Least Populous Countries in 2020', fontsize=14)
plt.gca().invert_yaxis() # Invert y-axis for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

