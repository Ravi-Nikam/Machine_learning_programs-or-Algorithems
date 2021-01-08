# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:55:05 2020

@author: Ravi Nikam
"""


# k nearest neighour (KNN)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('Social_Network_Ads.csv')

x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train= st.fit_transform(x_train)
x_test = st.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
# when we use matric = 'minkowski' eludian distance = 2 and manhattan distance =1
knn = KNeighborsClassifier(n_neighbors=15,metric='minkowski',p=2)
knn.fit(x_train,y_train)


y_predict=knn.predict(x_test)

# bcz we check accurancy after pridicion result with the acutual result 
from sklearn.metrics import confusion_matrix , classification_report
# give a accuracy and error rate
cnm = confusion_matrix(y_test,y_predict)
print(cnm)
# give precisoin and recall 
print(classification_report(y_test,y_predict))

from sklearn.model_selection import cross_val_score
accuracy_rate = []


# we find the mean of all the experiment
for i in range(1,40):
    # when we select metric as minkowski then p=1 means eculidin or manhattan
    knn = KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
    score=cross_val_score(knn,x,data['Purchased'],cv=10)
    # cross_val(model object,independent var,dependent vatr,no of experiment)
    # after perfoming 10 experiment we find the mean of it
    accuracy_rate.append(score.mean())


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
    score=cross_val_score(knn,x,data['Purchased'],cv=10)
    # in after we get mean of accuracy we perfome 1-score.mean() so we get error rate 
    # bcz after the finding accuracy remining are error rate
    error_rate.append(1-score.mean())


# we plot the graph
    # accuracy rate graph
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),accuracy_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
    plt.title('accuracy rate vs K value')
    plt.xlabel('k')
    plt.ylabel('accuracy rate')
    plt.show()
    
    
# error rate graph
    plt.figure(figsize=(10,6))
    plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
    plt.title('error rate vs K value')
    plt.xlabel('k')
    plt.ylabel('error rate')
    plt.show()
    
    
# ploting graph by clustring
    