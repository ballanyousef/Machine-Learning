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


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[5]:


dec_cls=DecisionTreeClassifier(max_depth=5)


# In[6]:


dec_cls.fit(X_train,y_train)


# In[7]:


y_pred=dec_cls.predict(X_test)


# In[8]:


from sklearn import metrics
print('Accuracy Score ',metrics.accuracy_score(y_test,y_pred))


# In[9]:


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

