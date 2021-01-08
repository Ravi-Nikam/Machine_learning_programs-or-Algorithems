# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:51:43 2020

@author: Ravi Nikam
"""
# Logistic Regression 
# Suv is purchsed or not 0 means not 1 means yes 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')
data.head()
x=data.iloc[:,[2,3]].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
x_train
# future scalling we apply (standardrization or normalization) method on it 
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
# fit to apply stand or nor both in one and then after we transform it
x_train=st.fit_transform(x_train)
x_train
# in transform we not apply any method we only transform it bcz it test data we didn't need to apply any method on it
x_test = st.transform(x_test)


# fit the logistic regression to the traning set 
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression(random_state=0)
lor.fit(x_train,y_train)

# we use to check the predicted value with the actual value
y_prdic=lor.predict(x_test)
# confusion matrix helpful for finding the model accurancy and Error of model
from sklearn.metrics import confusion_matrix
# we add confusion matrix in testing set bcz we have perfome operation on the final outcome
# check actual test value with the predicted value to finding the accuracy and error they check both
con = confusion_matrix(y_test,y_prdic)
# in output there is 11 error(8+3)=11 only and (65+24)= 89 best value of predication  
con

# now we plot a classification on graph

from matplotlib.colors import ListedColormap
X_set, y_set = x_train,y_train

# meshgrid function create a grid 
# X_set[row,column]
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min() - 1,stop=X_set[:,0].max() + 1  ,step=0.01),
                    np.arange(start=X_set[:,1].min() - 1,stop=X_set[:,1].max() + 1 , step=0.01))
plt.contourf(X1, X2, lor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, lor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()                                             