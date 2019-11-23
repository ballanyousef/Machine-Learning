#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv(r'C:\ML Data Source\Fremont_Bridge_Bicycle_Counter.csv',index_col='Date', parse_dates=True)
df.head()


# In[5]:


df.plot()
plt.show()


# In[14]:


df_timewise=df.groupby(df.index.time).mean()
hr_ticks=4*60*60*np.arange(6)
df_timewise.plot(xticks=hr_ticks)
plt.show()


# In[18]:


df_dayofweek=df.groupby(df.index.dayofweek).mean()
df_dayofweek.index=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
df_dayofweek.plot()
plt.show()

