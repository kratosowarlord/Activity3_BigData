#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# importing datastets
titanic_data=pd.read_csv(r"C:\Users\User\Desktop\p3\Titanic.csv")
test_data=pd.read_csv(r"C:\Users\User\Desktop\p3\test.csv")


# In[3]:


titanic_data.head


# In[4]:


titanic_data.describe


# In[5]:


print("Missing values in the data values")
total = titanic_data.isnull().sum().sort_values(ascending=False)
percent_1 = titanic_data.isnull().sum()/titanic_data.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data)


# ### Q1) Find out the overall chance of survival for a Titanic passenger.

# In[7]:


print("No. of passengers survived = ",titanic_data['survived'].value_counts()[1])
print("Passengers survival percentage = ",titanic_data['survived'].value_counts(normalize=True)[1]*100)


# ### Q2) Find out the chance of survival for a Titanic passenger based on their sex and plot it.

# In[8]:


sns.barplot(x="sex", y="survived", data=titanic_data)
print("Female passangers survived percentage = ", titanic_data["survived"][titanic_data["sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Male passangers survived percentage = ", titanic_data["survived"][titanic_data["sex"] == 'male'].value_counts(normalize = True)[1]*100)


# ### Q3) Find out the chance of survival for a Titanic passenger by traveling class wise and plot it.

# In[9]:


sns.barplot(x="pclass", y="survived", data=titanic_data)
print("P_Class-1 passangers survived percentage = ", titanic_data["survived"][titanic_data["pclass"] == 1].value_counts(normalize = True)[1]*100)
print("P_Class-2 passangers survived percentage = ", titanic_data["survived"][titanic_data["pclass"] == 2].value_counts(normalize = True)[1]*100)
print("P_Class-3 passangers survived percentage = ", titanic_data["survived"][titanic_data["pclass"] == 3].value_counts(normalize = True)[1]*100)


# ### Q4) Find out the average age for a Titanic passenger who survived by passenger class and sex. 

# In[10]:


fig = plt.figure(figsize=(12,5))
fig.add_subplot(121)
plt.title('Age/Sex per Survivors')
sns.barplot(data=titanic_data, x='pclass',y='age',hue='sex')


# In[11]:


meanAgeTrnMale = round(titanic_data[(titanic_data['sex'] == "male")]['age'].groupby(titanic_data['pclass']).mean(),2)
meanAgeTrnFeMale = round(titanic_data[(titanic_data['sex'] == "female")]['age'].groupby(titanic_data['pclass']).mean(),2)
print('Age MEAN per Sex')
print(pd.concat([meanAgeTrnMale, meanAgeTrnFeMale], axis = 1,keys= ['Male','Female']))


# ### Q5) Find out the chance of survival for a Titanic passenger based on number of siblings the passenger had on the ship and plot it.

# In[12]:


sns.barplot(x="sibsp", y="survived", data=titanic_data)
plt.title('Passangers Survival chance based on number of siblings the passenger')
print("SibSp 0 Survivors percentage = ", titanic_data["survived"][titanic_data["sibsp"] == 0].value_counts(normalize = True)[1]*100)
print("SibSp 1 Survivors percentage = ", titanic_data["survived"][titanic_data["sibsp"] == 1].value_counts(normalize = True)[1]*100)
print("SibSp 2 Survivors percentage = ", titanic_data["survived"][titanic_data["sibsp"] == 2].value_counts(normalize = True)[1]*100)


# ### Q6) Find out the chance of survival for a Titanic passenger based on number of parents/children the passenger had on the ship and plot it.

# In[13]:


sns.barplot(x="parch", y="survived", data=titanic_data)
plt.title('Survival chance based on parents/children')
plt.show()
print("Parch 0 Survivors percentage = ", titanic_data["survived"][titanic_data["parch"] == 0].value_counts(normalize = True)[1]*100)
print("Parch 0 Survivors percentage = ", titanic_data["survived"][titanic_data["parch"] == 1].value_counts(normalize = True)[1]*100)
print("Parch 0 Survivors percentage = ", titanic_data["survived"][titanic_data["parch"] == 2].value_counts(normalize = True)[1]*100)
print("Parch 0 Survivors percentage = ", titanic_data["survived"][titanic_data["parch"] == 3].value_counts(normalize = True)[1]*100)


# ### Q7) Plot out the variation of survival and death amongst passengers of different age.

# In[14]:


titanic_data["age"] = titanic_data["age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
titanic_data['agegroup'] = pd.cut(titanic_data['age'], bins, labels = labels)
sns.barplot(x="agegroup", y="survived", data=titanic_data)
plt.title('Age variation of survivor and dead passengers')
plt.show()


# In[15]:


g = sns.FacetGrid(titanic_data, col='survived')
g.map(plt.hist, 'age', bins=20)


# ### Q8) Plot out the variation of survival and death with age amongst passengers of different passenger classes.

# In[16]:


print("Age and Class variation of survivor and dead passengers")
grid = sns.FacetGrid(titanic_data, col='survived', row='pclass', size=3, aspect=2)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# ### Q9) Find out the survival probability for a Titanic passenger based on title from the name of passenger.

# In[17]:


combine = [titanic_data, test_data]
for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic_data['Title'],titanic_data['sex'])


# In[18]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

titanic_data[['Title', 'survived']].groupby(['Title'], as_index=False).mean()


# ### Q10) What conclusions are you derived from the analysis?
# ##### Please checkout the PDF file.

# In[ ]:




