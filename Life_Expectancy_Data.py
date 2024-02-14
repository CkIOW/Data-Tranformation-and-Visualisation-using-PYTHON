#!/usr/bin/env python
# coding: utf-8

# # importing necessary libraries for cleaning, transforming and visualizing data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # loading the dataset into a dataframe

# In[2]:


df = pd.read_csv('Life Expectancy Data.csv')


# In[3]:


df.head()


# # sanity check of the data(missing values,garbage values, duplicates) 

# In[4]:


#info of the data

df.info()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


#percentage of missing values

df.isnull().sum()/df.shape[0]*100


# In[8]:


df.duplicated().sum()


# In[9]:


#identifying the garbage values in the dataset

for i in df.select_dtypes(include="object").columns:
    print(df[i].value_counts())
    print("***"*10)


# # Exploratory Data Analysis

# In[10]:


df.describe()


# In[64]:


df.describe(include="object")


# In[11]:


import warnings
warnings.filterwarnings("ignore")


# In[15]:


#Histogram for each numeric columns

for i in df.select_dtypes(include="number").columns:
    sns.histplot(x=i, data=df)
    plt.show()


# In[65]:


#Boxplot to identify the outliers

for i in df.select_dtypes("number").columns:
    sns.catplot(x=i, kind='box', data=df)
    plt.show()


# In[20]:


df.select_dtypes(include="number").columns


# In[66]:


#Scatter plot to understand the relationship

for i in ['Year','Adult Mortality', 'infant deaths',
       'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
       ' thinness 5-9 years', 'Income composition of resources', 'Schooling']:
    sns.scatterplot(data=df, x=i, y='Life expectancy ')
    plt.show()


# In[29]:


#corellation with heatmap to interpret the relation and multicolliniarity

corelation = df.corr(numeric_only)


# In[33]:


plt.figure(figsize=(15,15))
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)


# In[48]:


df.isnull().sum()


# # Treating missing values

# In[44]:


#Filling the mean or median of the column to null values 

columns_to_fill = ["Polio", "Income composition of resources"]
for column in columns_to_fill:
    df[column].fillna(df[column].median(), inplace=True)


# In[47]:


df[" BMI "].fillna(df[" BMI "].median(), inplace=True)


# In[53]:


from sklearn.impute import KNNImputer
imputer=KNNImputer()


# In[54]:


#Imputing the null values with KNN

for i in df.select_dtypes(include="number").columns:
    df[i]=imputer.fit_transform(df[[i]])


# In[55]:


df.isnull().sum()


# # Treating the outliers

# In[57]:


def wisker(col):
    q1,q3=np.percentile(col,[25,75])
    iqr=q3-q1
    lw=q1-1.5*iqr
    uw=q3+1.5*iqr
    return lw,uw  


# df.columns

# In[58]:


df.columns


# In[60]:


for i in ['GDP','Total expenditure',' thinness 5-9 years',' thinness  1-19 years']:
    lw,uw=wisker(df[i])
    df[i]=np.where(df[i]<lw,lw,df[i])
    df[i]=np.where(df[i]>uw,uw,df[i])
    


# In[63]:


for i in ['GDP','Total expenditure',' thinness 5-9 years',' thinness  1-19 years']:
    sns.catplot(x=i,kind='box',data=df)
    plt.show()


# # Encoding of Data

# In[73]:


#labelEncoding and one hot encoding with pd.getdummies

dummy=pd.get_dummies(data=df,columns=["Country","Status"],drop_first=True)


# In[74]:


dummy


# In[ ]:




