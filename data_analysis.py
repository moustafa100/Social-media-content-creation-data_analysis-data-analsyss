#!/usr/bin/env python
# coding: utf-8

# Load the necessary packages

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import os
import pickle
import re


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[3]:


plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(8, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ## 1. Loading data

# In[4]:


content=pd.read_csv("Content (1).csv")
content.drop('Unnamed: 0',axis=1,inplace=True)
reaction_type=pd.read_csv("ReactionTypes (1).csv")
reaction_type.drop('Unnamed: 0',axis=1,inplace=True)
reactions=pd.read_csv("Reactions (1).csv")
reactions.drop('Unnamed: 0',axis=1,inplace=True)


# In[5]:


content.head()


# In[6]:


reactions.head()


# In[7]:


reaction_type.head()


# #### Combining two dataframes

# To create the model dataset, you want to start with the Reaction table as your base table. This 
# table shows all of the reactions to particular content IDs. To find out the category of these 
# pieces of content that have been reacted to, we must merge the Content table to the Reaction 
# table using a “left” join and merging on the “Content ID” column. Let’s call this new table “df”.
# 

# In[8]:


df=pd.merge(content,reactions, on='Content ID')


# Remove "URL" and "User ID_y" columns

# In[9]:


df.drop(['URL','User ID_y'],inplace=True,axis=1)


# Rename colunms name

# In[10]:


df.rename(columns={'User ID_x':'User ID','Type_x':'content_type','Type_y':'Type'},inplace=True)


# merging the ReactionTypes table onto “df” as a “left” join and merging on the column \
# which describes the reaction “Type”. Call this final dataset “A”.

# In[11]:


A=pd.merge(reaction_type,df,on='Type', how='left')


# In[12]:


A.head()


# #### Data types
# Often, it is useful to understand what data we are dealing with, as the data types might end up causing errors into our analysis at a later stage.
# Below, we can quickly see the dates in our dataset are not datetime types yet, which means we might need to convert them. In addition, we can see that the Score is full of integers so we can keep it in that form.
# 
# **Note**: We've transformed the output to a dataframe to facilitate visualization

# In[13]:


pd.DataFrame({"Data type":A.dtypes})


# In[14]:


# Transform date columns to datetime type
A["Datetime"]=pd.to_datetime(A["Datetime"],format='%Y-%m-%d')


# In[15]:


A.dtypes


# rename and reorder columns 

# In[16]:


A.rename(columns={'Type':'Reaction Type'},inplace=True)
A=A[["Content ID","Reaction Type","Datetime","User ID","content_type","Category","Sentiment","Score"]]
A.head()


# #### Missing data
# 
# We are also concerned we have a lot of missing data so we can check how much of our data is missing.\
# **Note**: We've transformed the output to a dataframe to facilitate visualization  **There are no missing value**

# In[17]:


pd.DataFrame({"Missing value":A.isnull().sum()})


# In[18]:


A["Category"].unique()


# In[19]:


def normalize_Category(x):
    
    if len(re.findall("dogs|animals|Animals|dogs", x))>0:
        return "animal"
    elif len(re.findall("Studying|education",x))>0:
        return "study"
    elif len(re.findall("healthy eating|food|veganism|fitness", x))>0:
        return "health"
    elif len(re.findall("technology|science", x))>0:
        return "tech"
    elif len(re.findall("tennis|Soccer", x))>0:
        return "sport"
    else :
        return "hobby"


# make 6 groups(animal,study,health,tech,sport and hobby)

# In[20]:


A["Category"].value_counts()
A["Category"]=A["Category"].apply(lambda x:normalize_Category(x)if pd.notnull(x) else x)


# In[21]:


A["Category"].unique()


# In[22]:


A["Reaction Type"].unique()


# how know month with most posts 

# In[49]:


post_year=A[["Content ID"]].set_index(A["Datetime"])


# In[50]:


year=post_year.groupby(pd.Grouper(freq='M')).size()
year.head(20)


# In[52]:


def line_format(label):
    """
    Convert time label to the format of pandas line plot
    """
    month=label.month_name()[:3]
    if label.month_name() == "January":
        month += f'\n{label.year}'
    return month
ax=year.plot(kind='bar',stacked=True,figsize=(18,10),rot=30)
ax.set_xticklabels(map(lambda x: line_format(x), year.index))
plt.xticks(fontsize=15)
plt.ylabel("Number of posts")
plt.show()


# In[27]:



sum_=A['Score'].groupby([A['Category']]).sum().sort_values(ascending=False)
sum_


# In[53]:


sum_.plot(kind="bar")
plt.ylabel("Aggregate popularity score")
plt.xlabel("categories")
plt.show()


# In[36]:


#number of posts per category
size_=A.groupby([A['Category']]).size().sort_values(ascending=False)
size_


# In[54]:


size_.sum()


# In[61]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Hobby', 'Health', 'Tech', 'Animal','Study','Sport'
sizes = [38, 23.6, 13.7, 13.2,5.8,5.7]
explode = (0.1, 0, 0, 0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Popularity percentage share from categories")
plt.show()


# In[32]:


size_.plot(kind="bar")


# In[ ]:




