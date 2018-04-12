#########################################################################
# File Name: predictValue.py
# Author: King
# mail: arturiapendragon_1@163.com
# Created Time: Tue 10 Apr 2018 02:23:53 PM CST
#########################################################################
#!/usr/bin/env python3
#! -*- coding: utf-8 -*-
import numpy as np
import scipy
import sympy as sum
import matplotlib.pyplot as plt
import pyecharts

data = np.load('homework.npz')

x =data['x'].reshape(-1,1)
d =data['d'].reshape(-1,1)
print(x)
print(d)




def funcA(x,a=2,b=1):
    return a*x+b


def fit(x,y,step=1000):
    y_temp=[]
    y_real=[]
    step=[]
    for i in range(1,len(x)-1):
        y_value =funcA(x[i,0])
        y_temp.append(y_value)
        y_real.append(y[i,0])
        step.append(i)
    return y_temp,y_real,step

y_predict,y_real,step =fit(x,d)




plt.plot(y_predict,'-o',color='r')
plt.plot(y_real,'-o',color='g')
plt.xlabel('x-value')
plt.ylabel('d-value')
plt.show()



from sklearn.model_selection import train_test_split
from sklearn import linear_model
x_train,x_test,y_train,y_test =train_test_split(x,d,test_size=0.3)

scores=[]






#************************************************************************
#from sklearn.model_selection import train_test_split
#from sklearn import linear_model

#x_train,x_test,y_train,y_test =train_test_split(x,d,test_size=0.3)


#regr=linear_model.LinearRegression()
#regr.fit(x_train,y_train)
#regr_y =regr.predict(x_test)
#print(regr_y)
#print(y_test)


#print(regr.coef_)
#print (regr.intercept_)
#print (regr.get_params)

#print(regr.score(x_train,y_train))

#************************************************************************

