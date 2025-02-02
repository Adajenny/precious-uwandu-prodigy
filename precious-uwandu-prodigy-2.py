#!/usr/bin/env python
# coding: utf-8

# ## Understanding the Titanic. Explore the Relationship between Varibales and Identify Patterns and Trends in the data.

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# In[9]:


titanic= pd.read_csv(r'C:\Users\pc\Documents\titanic.csv')
titanic


# # UNDERSTANDING THE TITANIC

# In[7]:


titanic.info()


# In[10]:


titanic.isna().sum()


# # HANDLING FOR MISSING VALUE

# In[16]:


# imput missing values for dataset
titanic['Embarked'] = titanic['Embarked'].replace({np.nan:'S'})


# In[18]:


titanic['Embarked'].isna().sum()


# In[24]:


titanic['Age'].isna().sum()


# In[26]:


titanic['Fare'].isna().sum()


# In[34]:


sns.boxplot(y=titanic['Age'].dropna(),x=titanic['Pclass'],palette = 'pastel', hue=titanic['Pclass'])


# In[56]:


# Define age replacements based on Pclass
age_replacement = {1: 38, 2: 29, 3: 27}

# Replace NaN values in 'Age' using 'Pclass'
titanic['Age'] = titanic.apply(
    lambda row: age_replacement[row['Pclass']] 
    if np.isnan(row['Age']) 
    else row['Age'],
    axis=1
)


# In[58]:


titanic['Age'].isna().sum()


# ## visualizations

# In[68]:


sns.countplot(x=titanic ['Pclass'], hue = titanic['Pclass'])


# In[72]:


sns.countplot(x =titanic['Survived'], hue= titanic['Sex'], palette ='viridis')


# In[11]:


sns.countplot(x = titanic['Survived'], hue = titanic['SibSp'])


# In[13]:


sns.countplot(x = titanic['Survived'], hue = titanic['Pclass'])


# In[15]:


maxAge = titanic['Age'].max()
minAge = titanic['Age'].min()
print(minAge, maxAge)


# In[17]:


bins = [0, 5, 12, 19, 35, 60, 100]
labels = ['Toddlers', 'Child', 'Teen', 'Young Adult', 'Adult', 'Old']
titanic['age_category'] = pd.cut(titanic['Age'], bins = bins, labels = labels)
titanic.head()


# In[19]:


sns.countplot(x = titanic['Survived'], hue = titanic['age_category'])


# In[21]:


numeric_data = titanic.select_dtypes(include = [np.number])
numeric_data


# In[23]:


numeric_data.corr()


# # Hypothesis Theory 

# ### 1. Relationship between Pclass anf fare

# H_O: There is no relatioship between Pclass and Fare. 
# 
# 𝐻_𝑎: There is a correlation between Pclass and Fare.
# 

# In[33]:


sns.histplot(x = titanic['Pclass'], bins = 20, kde = True, edgecolor = None, color = 'red')


# In[35]:


sns.histplot(x = titanic['Fare'], bins = 20, kde = True, edgecolor = None, color = 'red')


# In[37]:


sns.scatterplot(data = titanic, x = 'Age', y = 'Fare', hue = 'Pclass')


# In[39]:


correlation, p_value = stats.pearsonr(titanic['Pclass'], titanic['Fare'])
print(f'Correlation coefficient:{correlation}')
print(f'P_value:{p_value}')


# ##### alpha = 0.05

# In[42]:


if p_value <= 0.05:
    print("Reject the null hypothesis. Significant correlation.")
else:
    print("Fail to reject the null hypothesis. No significant correlation.")    
    


# ##### There is statistically significant but wear linear correlation

# #### 2. Test for class and Survival Association
# 

# Hypothesis:
# 
# 𝐻_0: Passenger class and survival are independent.
# 
# 𝐻_𝑎: Passenger class and survival are not independent.

# In[50]:


from scipy.stats import chi2_contingency


# In[52]:


contingency_table = pd.crosstab(titanic['Pclass'], titanic['Survived'])
contingency_table


# In[54]:


chi2, p, dof, expected = chi2_contingency(contingency_table)


# In[56]:


print("Expected Frequencies:")
print(expected)


# #### No frequency is less than 5 therfore Chi_square analysis is valid

# In[60]:


if p < 0.05:
    print('Null hypothesis is rejected as their is prove of association')
else:
    print('Null hypothesis is failed to reject')


# ##### There are some Relationship between the class and survived

# #### 3. Proportion Comparison Across Groups: survival rates across embarkation ports

# Hypothesis:  
# H_0: Survival rates are the same across embarkment ports  
# H_a: Survival rates are not the same across embarkment ports

# In[89]:


contingency_table2 = pd.crosstab(titanic['Embarked'], titanic['Survived'])
contingency_table2


# In[91]:


chi2, p2, dof, expected = chi2_contingency(contingency_table2)


# In[93]:


print("Expected Frequencies:")
print(expected)


# ##### No frequency is less than 5 therfore Chi_square analysis is valid¶

# In[98]:


if p2 < 0.05:
    print('Null hypothesis is rejected as their is prove of association')
else:
    print('Null hypothesis is failed to reject')


# In[ ]:




