#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Import Libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df=pd.read_csv(r'C:\ML Data Source\diabetes.csv')
df.head()


# In[3]:


X=df.drop(['Outcome'], axis=1)
y=df['Outcome']


# In[4]:


dec_cls=DecisionTreeClassifier(max_depth=5,criterion='entropy')


# In[5]:


dec_cls.fit(X,y)


# In[6]:


y_pred=dec_cls.predict(X)


# In[7]:


from sklearn import metrics
print('Accuracy Score ',metrics.accuracy_score(y,y_pred))


# In[8]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  


dot_data = StringIO()
export_graphviz(dec_cls, out_file='tree.dot',  
                filled=True, rounded=True,
                special_characters=True,feature_names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'], class_names=['0','1'])


# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

