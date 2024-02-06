#!/usr/bin/env python
# coding: utf-8

# # Problem statement: Flight price prediction model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv('Flight_Booking.csv')
df


# In[3]:


# EDA -Exploratory data analysis


# In[4]:


df.info()


# In[5]:


df.drop(['Unnamed: 0','flight'],axis=1,inplace=True)


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.duplicated().sum()


# In[10]:


#Visualization


# In[11]:


plt.figure(figsize=(6,3))
sns.barplot(data=df,x="airline",y="price",hue="class")
plt.show()


# In[16]:


plt.figure(figsize=(6,3))
sns.lineplot(data=df,x="days_left",y="price",hue="airline")
plt.show()


# In[15]:


plt.figure(figsize=(6,3))
sns.barplot(data=df,x="airline",y="price",hue="departure_time")
plt.show()


# In[23]:


plt.figure(figsize=(10,5))
sns.barplot(data=df,x='stops',y='price',hue='airline')
plt.show()


# In[24]:


# Data Pre-processing
# Label Encoding


# In[25]:


from sklearn.preprocessing import LabelEncoder


# In[31]:


le=LabelEncoder()


# In[32]:


df.columns


# In[33]:


for col in df.columns:
    if df[col].dtype=="object":
        df[col]=le.fit_transform(df[col])


# In[34]:


df


# In[35]:


# VIf for the Feature selection


# In[36]:


from statsmodels.stats.outliers_influence import  variance_inflation_factor


# In[37]:


col_list=[]
for i in df.columns:
    if (df[i].dtype!="object") & (i!="price"):
        col_list.append(i)


# In[38]:


col_list


# In[39]:


x=df[col_list]
x


# In[40]:


vif_data=pd.DataFrame()
vif_data["features"]=x.columns
vif_data["VIF values"]=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
vif_data


# In[41]:


#Feature scaling


# In[42]:


# standardization


# In[45]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[47]:


x=sc.fit_transform(x)
x


# In[50]:


x=pd.DataFrame(x,columns=col_list)
x


# In[53]:


x #will be my Independent data
y=df["price"] #dependent
y


# In[54]:


# let's split the data into training and Testing
from sklearn.model_selection import train_test_split


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=52)


# In[56]:


x_train


# In[60]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# # LinearRegression

# In[61]:


lr_model=LinearRegression()


# In[62]:


lr_model


# In[63]:


# training the model 
lr_model.fit(x_train,y_train)


# In[64]:


# testing model
lr_pred=lr_model.predict(x_test)


# In[65]:


lr_pred


# In[66]:


y_test


# In[67]:


# Evaluation of the model
from sklearn.metrics import *


# In[68]:


lr_score=r2_score(y_test,lr_pred)
lr_score


# In[69]:


lr_rmse=np.sqrt(mean_squared_error(y_test,lr_pred))
lr_rmse


# # DecisionTreeRegression

# In[70]:


dt_model=DecisionTreeRegressor()


# In[71]:


dt_model


# In[73]:


dt_model.fit(x_train,y_train)


# In[74]:


dt_pred=dt_model.predict(x_test)
dt_pred


# In[75]:


y_test


# In[76]:


dt_score=r2_score(y_test,dt_pred)
dt_score


# In[84]:


dt_rmse=np.sqrt(mean_squared_error(y_test,dt_pred))
dt_rmse


# # RandomForestRegressor

# In[78]:


rf_model=RandomForestRegressor()


# In[79]:


rf_model.fit(x_train,y_train)


# In[80]:


rf_pred=rf_model.predict(x_test)
rf_pred


# In[81]:


rf_score=r2_score(y_test,rf_pred)
rf_score


# In[85]:


rf_rmse=np.sqrt(mean_squared_error(y_test,rf_pred))
rf_rmse


# In[86]:


# Conclusion:
print("Linear Regression- r2_score=",lr_score,"and rmse=",lr_rmse)
print("DT regressor-  r2_score=",dt_score,"and rmse=",dt_rmse)
print("RF regressor-  r2_score=",rf_score,"and rmse=",rf_rmse)


# In[ ]:




