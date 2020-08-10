#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[49]:


train_data=pd.read_csv('C:/Users/HP/Desktop/Kaggle/train.csv')


# In[50]:


train_data.head()


# In[51]:


test_data=pd.read_csv('C:/Users/HP/Desktop/Kaggle/test.csv')


# In[52]:


test_data.head()


# In[53]:


test_data.info()


# In[54]:


train_data.info()


# In[55]:


train_data['Age']=train_data['Age'].fillna(value=train_data['Age'].mean())


# In[56]:


train_data.info()


# In[57]:


X=train_data.drop(columns=['Survived','Name','Ticket','Cabin','Embarked'])
X.head()


# In[58]:


X.shape


# In[59]:


y=train_data.Survived
y


# In[60]:


y.shape


# In[61]:


from sklearn.preprocessing import LabelEncoder


# In[62]:


le=LabelEncoder()
X['Sex']=le.fit_transform(X['Sex'])


# In[63]:


X.head()


# In[64]:


from sklearn.model_selection import train_test_split


# In[104]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# In[105]:


#Logistic Regression


# In[106]:


from sklearn.linear_model import LogisticRegression


# In[107]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[108]:


predictions=logmodel.predict(X_test)


# In[109]:


predictions


# In[110]:


from sklearn.metrics import classification_report


# In[111]:


print(classification_report(y_test,predictions))


# In[112]:


logmodel.score(X_test,y_test)


# In[113]:


test_data1=test_data.drop(columns=['Name','Ticket','Cabin','Embarked'])
test_data1


# In[114]:


from sklearn.preprocessing import LabelEncoder


# In[115]:


le=LabelEncoder()
test_data1['Sex']=le.fit_transform(test_data1['Sex'])


# In[116]:


test_data1.head()


# In[117]:


test_data1['Age']=test_data1['Age'].fillna(value=test_data1['Age'].mean())


# In[118]:


test_data1['Age']


# In[127]:


test_data1['Fare']=test_data1['Fare'].fillna(value=test_data1['Fare'].mean())


# In[128]:


test_data1['Fare']


# In[129]:


test_data1.head()


# In[130]:


test_data1


# In[131]:


test_data1.info()


# In[132]:


pred_final=logmodel.predict(test_data1)


# In[134]:


pred_final


# In[136]:


output=pd.DataFrame({'PassengerId':test_data1.PassengerId,'Survived':pred_final})
output.to_csv(r'C:\Users\HP\Desktop\Kaggle\my_submission_logreg.csv',index=False)

