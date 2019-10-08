#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r'C:\Sanrusha-Canon Laptop\Udemy\Machine Learning\SampleDataSet\bank-additional\bank-additional.csv', delimiter=';')
df.head()


# In[3]:


df.shape


# In[4]:


y=df['y']
X=df.drop(['y'], axis=1)


# In[5]:


from sklearn.preprocessing import LabelEncoder
lbc=LabelEncoder()
for i in range(len(X.columns)):
    X.iloc[:,i]=lbc.fit_transform(X.iloc[:,i])


# In[6]:


X.head()


# In[7]:


from sklearn.preprocessing import StandardScaler
stdscl=StandardScaler()
X=stdscl.fit_transform(X)


# In[8]:


df.corr()


# In[9]:


pd.DataFrame(X).head()


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[11]:


#train the model
from sklearn.linear_model import LogisticRegression
lrc=LogisticRegression()
lrc.fit(X_train,y_train)


# In[12]:


#test the model and predict
y_pred=lrc.predict(X_test)


# In[13]:


from sklearn import metrics
print("Accuracy ", metrics.accuracy_score(y,lrc.predict(X)))


# In[15]:


df3=pd.DataFrame({'Actual':y.values,'Predicted':lrc.predict(X)})
df3.head(15)


# In[16]:


j=0
for i in range(len(df3)):
    if df3.iloc[i,0]!=df3.iloc[i,1]:
        j=j+1

print(j)

