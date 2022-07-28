#!/usr/bin/env python
# coding: utf-8

# # COVID IN INDIA
# 
# 

# In[ ]:





# In[55]:


# Basic Packages
import pandas as pd  #read the dataset 
import numpy as np   #numerical python or array or matrix mulptiple
import matplotlib.pyplot as plt  #plot the graph
import seaborn as sns   #graphical representation


# In[7]:


covid= pd.read_csv('covid_19_india.csv')


# In[8]:


covid.head()


# In[14]:


covid.shape


# In[15]:


covid.isnull()


# In[16]:


# to find any null value in between use 
covid.isnull().sum()


# In[17]:


# just checking the information
covid.info()


# In[27]:


covid['Confirmed'].sum()


# In[28]:


covid['Deaths'].sum()


# In[30]:


#calculate the total mortality rate which is the death_sum/confirmed cases
mortality=(covid['Deaths'].sum()/covid['Confirmed'].sum())
mortality
    


# In[68]:


#us now have a look at the most recent records for each state to gain an idea about where we stand currently. 
#From the last set of records, we can see that we have data till 25th April 2020
covid_latest = covid[covid['Date']=="25/04/20"]
covid_latest.head()


# In[69]:


covid_latest['Confirmed'].sum()


# In[70]:


covid_latest['Deaths'].sum()


# # Statewise Figure

# In[71]:


#the most number of inspected cases as of now,in which state
covid_latest = covid_latest.sort_values(by=['Confirmed'], ascending = False)
plt.figure(figsize=(12,8), dpi=80)
plt.bar(covid_latest['State/UnionTerritory'][:5], covid_latest['Confirmed'][:5],
        align='center',color='lightgrey')
plt.ylabel('Number of Confirmed Cases', size = 12)
plt.title('States with maximum confirmed cases', size = 16)
plt.show()


# In[72]:


# which states have the most deaths
covid_latest = covid_latest.sort_values(by=['Deaths'], ascending = False)
plt.figure(figsize=(12,8), dpi=80)
plt.bar(covid_latest['State/UnionTerritory'][:5], covid_latest['Deaths'][:5], align='center',color='lightgrey')
plt.ylabel('Number of Deaths', size = 12)
plt.title('States with maximum deaths', size = 16)
plt.show()


# In[81]:


# making graph for a particular state
df=covid.loc[(covid['State/UnionTerritory']=='Kerala')]


# In[82]:


df.head()


# In[83]:


df.shape


# In[84]:


#graph
sns.countplot(x='ConfirmedIndianNational',data=df)


# In[22]:


get_ipython().system(' pip install plotly')


# In[10]:


import plotly.offline as py
import plotly.graph_objs as go


# In[11]:


Cured_chart=go.Scatter(x=df['Date'],y=df['Cured'],name='Cured Rate')
Deaths_chart=go.Scatter(x=df['Date'],y=df['Deaths'],name='Deaths Rate')
py.iplot([Cured_chart,Deaths_chart])


# # SVM

# In[6]:


get_ipython().system('pip install sklearn')


# In[33]:


from sklearn.svm import LinearSVR


# In[34]:


dfc=df[['Confirmed']]
dfc=dfc.values


# In[35]:


# train and test


# In[36]:


train_size=int(len(dfc)*0.80)
test_size=len(dfc)-train_size


# In[37]:


train,test=dfc[0:train_size,:],dfc[train_size:len(dfc),:]


# In[38]:


test.shape


# In[39]:


def create(dataset,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)


# In[40]:


look_back=2
trainX,trainY=create(train,look_back=look_back)
testX,testY=create(train,look_back=look_back)


# In[41]:


trainX


# In[42]:


model=LinearSVR()


# In[43]:


model.fit(trainX,trainY)


# In[44]:


predict1=model.predict(testX)


# In[45]:


plt.plot(testY,color='red',label='Actual Values')
plt.plot(predict1,color='blue',label='Predicted Values')
plt.ylabel('meantemp')
plt.legend()


# In[48]:


df=pd.DataFrame({'Actual':testY.flatten(),'predicted':predict1.flatten()})
df


# In[49]:


#bar graph
df.plot(kind='bar',figsize=(16,10))


# In[ ]:




