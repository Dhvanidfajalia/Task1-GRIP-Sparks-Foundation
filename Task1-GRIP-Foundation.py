#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[4]:


# STEP1: IMPORT LIBRARIES


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# %matplotlib inline
import statsmodels.api as sm 


# In[6]:


#STEP2: DATASET COLLECTION

url="http://bit.ly/w-data"
dataset=pd.read_csv(url)
dataset


# In[7]:


#STEP 3: EXPLORING NATURE OF THE DATA

dataset.shape


# In[8]:


dataset.head()


# In[9]:


dataset.describe()


# In[10]:


# STEP 4: CREATING REGRESSION

y=dataset['Scores']
x=dataset['Hours']


# In[15]:


# STEP 5: EXPLOREING DATA

plt.scatter(x,y)
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Percentage',fontsize=15)
plt.grid()
plt.show()


# In[17]:


# STEP 6: REGRESSION

x1=sm.add_constant(x)
results=sm.OLS(y,x1).fit()   #results has the output of OLS Regression
results.summary()


# In[19]:


# STEP 7: APPLY OUTPUT TO EQUATION AND RETURN THE CODE

plt.scatter(x,y)
yhat= 9.7758*x+2.4837
fig=plt.plot(x,yhat,lw=5, c="red",label='regression line')
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Scores',fontsize=15)
plt.grid()
plt.show()


# In[20]:


# STEP 8: CORRELATION

dataset.corr(method='pearson')


# In[21]:


dataset.corr(method='spearman')


# In[27]:


# STEP 9: LINEAR REGRESSION

# using iloc function for dividing the data 

X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,:1].values


# In[28]:


X


# In[29]:


Y


# In[31]:


# splitting data into training and testing data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[41]:


# training the model

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, Y_train)


# In[42]:


# STEP 10: VISUALIZING TRAINING DATA

line = model.coef_*X + model.intercept_

# plotting for the training data
plt.rcParams["figure.figsize"] = [14,7]
plt.scatter(X_train, Y_train, color='orange')
plt.plot(X, line, color='blue');
plt.xlabel('Hours Studied',fontsize=20)
plt.ylabel('Percentage Score',fontsize=20)
plt.grid()
plt.show()


# In[43]:


# plotting for the testing data

plt.rcParams["figure.figsize"] = [14,7]
plt.scatter(X_test, Y_test, color='orange')
plt.plot(X, line, color='blue');
plt.xlabel('Hours Studied',fontsize=20)
plt.ylabel('Percentage Score',fontsize=20)
plt.grid()
plt.show()


# In[65]:


# STEP 11: PREDICTIONS

# making predictions

print(X_test)
y_pred = model.predict(X_test)


# In[46]:


# camparing actual and predicted

Y_test # actual data


# In[48]:


y_pred # predicted data


# In[49]:


comp = pd.DataFrame({ 'Actual':[Y_test], 'Predicted':[y_pred] })
comp


# In[70]:


# STEP 12: SCORE PREDICTION FOR 9.25 HRS/DAY

# testing with own data

hours = 9.25
#hours = np.array(hours).reshape(1,-1)
own_pred=model.predict([[hours]])
print("Total hours is", format(hours))
print("Total score is", format(own_pred[0]))


# In[ ]:




