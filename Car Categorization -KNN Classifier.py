#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn import datasets

df=pd.read_csv(r'<your directory>\car.data',header=None,delimiter=',',names=['buying','maint','doors','persons','lug_boot','safety','CAR'])
df.head()


# In[3]:


from sklearn.preprocessing import LabelEncoder
lbc=LabelEncoder()
df["buying"]=lbc.fit_transform(df["buying"])
df["maint"]=lbc.fit_transform(df["maint"])
df["lug_boot"]=lbc.fit_transform(df["lug_boot"])
df["safety"]=lbc.fit_transform(df["safety"])
df["doors"]=lbc.fit_transform(df["doors"])
df["persons"]=lbc.fit_transform(df["persons"])
#df["CAR"]=lbc.fit_transform(df["CAR"])
df.head()


# In[4]:


df['CAR'].unique()


# In[5]:


X=df.drop(['CAR'], axis=1).values
y=df['CAR'].values


# In[6]:


sns.pairplot(df,x_vars=['buying','maint','doors','persons','lug_boot','safety'], y_vars="CAR",size=3.0)
plt.show()


# In[7]:


df.applymap(np.isreal).head()


# In[8]:


#Split training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error
y1=lbc.fit_transform(df["CAR"])
rmse=[]
for k in range(20):
    k=k+1
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y1)
    #y_pred_knn=knn.predict(X)
    rmse.append(sqrt(mean_squared_error(y1,knn.predict(X))))
    print('K value ',k,'rmse ',sqrt(mean_squared_error(y1,knn.predict(X))))


# In[10]:


knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)


# In[11]:


from sklearn import metrics
print("Accuracy ",metrics.accuracy_score(y,knn.predict(X)))


# In[12]:



import matplotlib.pyplot as plt
import numpy as np

for xe, ye in zip(X, y):
    plt.scatter(xe,[ye] * len(xe),color="blue",marker="o",s=100)
    
for xe, ye in zip(X, knn.predict(X)):
    plt.scatter(xe,[ye] * len(xe),color="yellow",marker="*",s=10)

plt.show()

