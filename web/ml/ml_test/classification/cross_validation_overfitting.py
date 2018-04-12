"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
#coding: UTF-8
#Definiation of COLs:
#1. sepal length in cm
#2. sepal width in cm
#3. petal length in cm
#4. petal width in cm
#5. class:
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica
#Missing Attribute Values: None

from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

iris = load_iris()
#print(iris.data)
#print(iris.target)
X = iris.data
y = iris.target
#print(X, y)
# (0) feature engineering
# the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
poly = PolynomialFeatures(2)
X_Poly = poly.fit_transform(X)
#X = X_Poly
#print(X_Poly)


# (1) test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)

# (2) Model training
knn.fit(X_train, y_train)

# (3) Predict & Estimate the score
y_pred = knn.predict(X_test)
#print("y_test")
#print(y_test)
#print("y_pred")
#print(y_pred)
print(knn.score(X_test, y_test))

#over fiting

from sklearn.learning_curve import learning_curve#学习曲线模块
from sklearn.learning_curve import validation_curve #validation_curve模块
import matplotlib.pyplot as plt
import numpy as np

#*************************************
#建立参数测试集
#train_sizes,train_loss,test_loss= learning_curve(KNeighborsClassifier( n_neighbors=13),X,y, cv=15,scoring='mean_squared_error',train_sizes=[0.1,0.25,0.5,0.75,1])

#*************************************
#建立参数测试集
#param_range = np.logspace(-6, -2.3, 5)
train_sizes =range(1,30)

#使用validation_curve快速找出参数对模型的影响
train_loss, test_loss = validation_curve(KNeighborsClassifier(), X, y, param_name='n_neighbors', param_range=train_sizes, cv=10, scoring='mean_squared_error')


#*************************************
#建立参数测试集
#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color = 'r',label='training')
plt.plot(train_sizes,test_loss_mean,'o-',color = 'g',label='cross_validation')

plt.xlabel('traing examples')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()



