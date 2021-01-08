# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:24:56 2020

@author: Ravi Nikam
"""

# Decision tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# below data is non linear 
data=pd.read_csv('Position_Salaries.csv')
#data=pd.read_csv('marji.csv')

x=data.iloc[:,1:2].values
y=data.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)


# bcz of small dataset we can't split into traning and testing
# and we not need FS 


# based on Entropy and gain it will split all range of your independent variable into interval
# Red point is your interval avg
# all line cutting ------- is intervalx
x_re=np.arange(min(x),max(x),0.01)
x_re=x_re.reshape(len(x_re),1)
plt.scatter(x,y,color="Red")
plt.title('Decision Tree Regression')
plt.xlabel('Lavel')
plt.ylabel('Salary')
plt.plot(x_re,regressor.predict(x_re),color="Blue")
plt.show()

regressor.predict([[2]])





