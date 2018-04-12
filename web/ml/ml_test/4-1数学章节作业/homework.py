#########################################################################
# File Name: Desktop/4-1数学章节作业/homework.py
# Author: King
# mail: arturiapendragon_1@163.com
# Created Time: Tue 10 Apr 2018 02:41:18 PM CST
#########################################################################
#!/usr/bin/env python3
#! -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#获取样本
data = np.load('homework.npz')
x = data['x']
d = data['d']
#假设样本存在的关系
#y=2*x+1

#损失函数 (y-d)**2 尽可能小
def loss(a,b,c,x,d):
    return (a*x**2+b*x+c-d)**2

def grad(a,b,c,x,d):
    ga = 2*(a*x**2+b*x+c-d)*x**2
    gb = 2*(a*x**2+b*x+c-d)*x
    gc = 2*(a*x**2+b*x+c-d)
    return ga,gb,gc

def predict(a,b,c):
    return a*x**2+b*x+c
ai = 0.5
bi = 0.5
ci = 0.5

for i in range(1000):
    idx = np.random.randint(300)
    sx = x[idx]
    sd = d[idx]
    ga,gb,gc=grad(ai,bi,ci,sx,sd)
    ai -= ga*0.01
    bi -= gb*0.01
    ci -= gc*0.01
    print(ai,bi,loss(ai,bi,ci,sx,sd))


plt.scatter(x,d,color='r')
plt.scatter(x,predict(1,-2,0.3))
plt.show()

