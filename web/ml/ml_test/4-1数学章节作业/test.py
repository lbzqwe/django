
# coding: utf-8

# In[7]:


import numpy as np
import scipy
import sympy as sym
import matplotlib
import numpy.random
import time
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
sym.init_printing()

data=np.load("homework.npz")
#x1=data['x'].reshape(-1,1)
#x=np.insert(x1, 0, values=1, axis=1)
#y=data['d'].reshape(-1,1)
orig=np.column_stack((np.ones(len(data['x'])),data['x'],data['d']))
cols= orig.shape[1]
x = orig[:,0:cols-1]
y = orig[:,cols-1:cols]
theta=np.zeros([1,2])
n=50
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(x, theta):
    return sigmoid(np.dot(x, theta.T))
def cost(x, y, theta):
    left = np.multiply(-y, np.log(model(x, theta)))
    right = np.multiply(1 - y, np.log(1 - model(x, theta)))
    return np.sum(left - right) / (len(x))
def gradient(x, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(x, theta)- y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, x[:,j])
        grad[0, j] = np.sum(term) / len(x)

    return grad
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2

def stopCriterion(type, value, threshold):
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    x = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return x, y
def descent(data, theta, batchSize, stopType, thresh, alpha):
    #梯度下降求解

    init_time = time.time()
    i = 0
    k = 0
    x, y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(x, y, theta)]


    while True:
        grad = gradient(x[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize
        if k >= n:
            k = 0
            x, y = shuffleData(data)
        theta = theta - alpha*grad
        costs.append(cost(x, y, theta))
        i += 1

        if stopType == STOP_ITER:       value = i
        elif stopType == STOP_COST:     value = costs
        elif stopType == STOP_GRAD:     value = grad
        if stopCriterion(stopType, value, thresh): break

    return theta, i-1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==n: strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta


# In[ ]:


runExpe(orig, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)


# In[14]:


#标准化数据试一试
from sklearn import preprocessing as pp

scaled_data = orig.copy()
scaled_data[:, 0:2] = pp.scale(orig[:, 0:2])


# In[15]:




