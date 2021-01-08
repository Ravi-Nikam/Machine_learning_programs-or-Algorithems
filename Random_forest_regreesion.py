# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:16:07 2020

@author: Ravi Nikam
"""
# Random Forest Regression

import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('Position_Salaries.csv')

x = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x_train,y_train)


#without spliting
regressor2 = RandomForestRegressor(n_estimators=300,random_state=0)
regressor2.fit(x,y)


regressor.predict([[6.5]])

#without spliting
regressor2.predict([[6.5]])


# check with traning and testing data below traning data is less so we can't get 
# more accuracy
import numpy as np
x_grid=np.arange(min(x_train),max(x_train),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x_train,y_train,color="red")
plt.title('Random Forest learning with traning and testing set')
plt.xlabel('Lavel')
plt.ylabel('Salary')
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.show()




# bcz of small dataset we cannot take traning and testing dataset
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color="red")
plt.title('Random Forest learning without traning and testing set')
plt.xlabel('Lavel')
plt.ylabel('Salary')
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.show()
