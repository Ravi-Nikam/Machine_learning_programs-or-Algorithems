# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:01:18 2020

@author: Ravi Nikam
"""

# multiple Linear Regression

import pandas as pd
import numpy as np

data=pd.read_csv('Startup_Data.csv')
data.head()
# split independent and dependent variable

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encode = LabelEncoder()
x[:,3]=label_encode.fit_transform(x[:,3])

ct = ColumnTransformer(
        [('encoder',OneHotEncoder(categories='auto',dtype=np.int),[3])],
        remainder='passthrough'
    )
x = np.array(ct.fit_transform(x))

# Avoidng the dummy variable trap
x = x[:,1:]

# spliting and test data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Train the dataset
from sklearn.linear_model import LinearRegression
# this library LinearRegression automatically add the b0 as constant
linear_regression = LinearRegression()
linear_regression.fit(x_train,y_train)

# predicting the test set result
x_predi=linear_regression.predict(x_test)

# now we are going to use BackwardElimination method
# the statsmodels library can't add directly b0 as constant so we have to add it manually
import statsmodels.api  as sm
# add the x0 as 1
# we add intercept manually
x = np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
# now we start Backward Elimination
# we create new matrix of feature contain all feature which is defined below
x_opt = x[:,[0,1,2,3,4,5]]
x_opt=x_opt.astype(float)
regresion_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regresion_OLS.summary()

x_opt = x[:,[0,1,3,4,5]]
x_opt=x_opt.astype(float)
regresion_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regresion_OLS.summary()


x_opt = x[:,[0,3,4,5]]
x_opt=x_opt.astype(float)
regresion_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regresion_OLS.summary()


x_opt = x[:,[0,3,5]]
x_opt=x_opt.astype(float)
regresion_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regresion_OLS.summary()


# so thare is R&D is very powerful variable for predication
x_opt = x[:,[0,3]]
x_opt=x_opt.astype(float)
regresion_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regresion_OLS.summary()