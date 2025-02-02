#!/usr/bin/env python
# coding: utf-8

# ## Building a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### performing a Cross-Validation Score

# Let's import model_selection from the module cross_val_score.

# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.metrics import accuracy_score, classification_report as cr
from sklearn.preprocessing import StandardScaler
from sklearn import tree


# In[42]:


bank_data= pd.read_csv(r"C:\Users\pc\AppData\Local\Microsoft\Windows\INetCache\IE\A9U9WLPK\Bank_Data[1].csv")
bank_data


# In[11]:


bank_data.isna().sum() #using the isna function to check for missing value dataset


# ### Convert categorical to continous variable

# In[14]:


def labeltransformer(df, column):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])


# In[16]:


labeltransformer(bank_data, 'job')
labeltransformer(bank_data, 'marital')
labeltransformer(bank_data, 'education')
labeltransformer(bank_data, 'default')
labeltransformer(bank_data, 'housing')
labeltransformer(bank_data, 'loan')
labeltransformer(bank_data, 'contact')
labeltransformer(bank_data, 'month')
labeltransformer(bank_data, 'poutcome')
labeltransformer(bank_data, 'y')


# In[18]:


bank_data.sample(7)


# ### split features independent and target variables

# In[21]:


x = bank_data.drop('y', axis = 1).values
y = bank_data['y'].values


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[27]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)


# #### Training

# In[44]:


tree_model = tree.DecisionTreeClassifier()
tree_model.fit(x_train, y_train)


# #### Testing

# In[47]:


y_predict = tree_model.predict(x_test)
print(cr(y_test, y_predict))


# In[49]:


tree_model.predict([[50, 4, 1, 1, 1, 19, 0, 0, 0, 14, 10, 59, 1, -1, 0, 2]])


# In[ ]:




