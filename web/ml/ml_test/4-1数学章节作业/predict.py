#########################################################################
# File Name: predict.py
# Author: lbz
# mail: 120498304@qq.com
# Created Time: Wed 11 Apr 2018 12:23:07 PM CST
#########################################################################
#!/usr/bin/env python3
#! -*- coding: utf-8 -*-
import numpy as np
import scipy
import sympy as sym
import matplotlib.pyplot as plt
sym.init_printing()

x1=np.random.random(1000)
x2=np.random.random(1000)*5

#nomalization
x1=x1/np.std(x1)
x2=x2/np.std(x2)

print(np.mean(x1))
print(np.std(x1),np.std(x2))

plt.scatter(x1,x2)
plt.show()

#********************************************

rand1=np.random.normal(loc=1,scale=3,size=[1000])*10
rand2=rand1 *2+0.1*np.random.normal(loc=1,scale=3,size=[1000])*10

#E((X-E(X)*(Y-E(Y))
def Conv(x,y):
    return np.mean((x-np.mean(x))*(y-np.mean(y)))

print( Conv(rand1,rand2)/np.std(rand1)/np.std(rand2))


