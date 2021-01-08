# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:50:12 2020

@author: Ravi Nikam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
 
data=pd.read_csv('Position_Salaries.csv')

# spliting the data

x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

# bcz of small dataset we are not split into testing and traing dataset

# not need for future scalling bcz we are using linerregression which does FS itself

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures

# generated matrix x is transform into polynomialFeature matrix x_poly
poly_reg = PolynomialFeatures(degree=4)
x_poly=poly_reg.fit(x)
x_poly=poly_reg.transform(x)
# now we build secound object of  LinearRegression we fit our new  matrix x_poly
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


# now we make simple and polynomial chart 

# below graph in we check our orignal value in linear chart 
# with we check our predicated line 
# apd predict krli line plt ni ana pramane Salary Ketli che linear chart throw
plt.scatter(x,y,color="Red")
plt.plot(x,lin_reg.predict(x),color="B")
plt.title("Truth or Buff(Linear regression)")
plt.xlabel("Lavel")
plt.ylabel("Salary")
plt.show()


# below graph in we check our orignal value with polynomial chart 
# with we check our predicated line 
# apd predict krli line plt ni ana pramane Salary Ketli che polynomial chart throw

plt.scatter(x,y,color="Red")
# lin_reg2.predict(poly_reg.fit_transform(x))
plt.plot(x,lin_reg2.predict(x_poly),color="green")
plt.title("Truth or Buff(polynomial regression)")
plt.xlabel("Lavel")
plt.ylabel("Salary")
plt.show()
# In the above code, we have taken lin_reg2.predict(poly_reg.fit_transform(x), instead of x_poly, because we want a Linear regressor object to predict the polynomial features matrix.


# now we are find the person predication data lavel 6.5 then salary is ??

print("1 for linear regression we find the salary of those person")
lin_reg.predict([[6.5]])
print("2 for polynomial regression we find the salary of those person")
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
