import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import io
import sys
from sklearn import preprocessing
from sklearn.svm import SVC

import matplotlib.pyplot as plt
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
# 特征工程构造特征方法：将数字1,2,3 ... 等构造特征（重要影响因素，对预测number, "fizz", "buzz", "fizzbuzz"有帮助的因素），构造为三个维度。
# 将每个输入的数，表示为一个特征数组（向量），这个特征数组有三个维度。
def feature_engineer(i):

    return np.array([i % 3, i % 5, i % 15])

# 将需要预测的指标转换为数字方法：将数据的真实值（预测结果）number, "fizz", "buzz", "fizzbuzz"
# 分别对应转换为数字 3, 2, 1, 0，这样后续能被计算机处理
def construct_sample_label(i):
    if   i % 15 == 0:
        return np.array(3)
    elif i % 5  == 0:
        return np.array(2)
    elif i % 3  == 0:
        return np.array(1)
    else:
        return np.array(0)

def output(i,value):
    if i == 1:
        return 'fizz'
    elif i == 2:
        return 'buzz'
    elif i == 3:
        return 'fizzBuzz'
    elif i == 0:
        return str(value)
    else:
        return 'no classify'
#[1, 1] [2, 2] [3, fizz]
# 生成训练集和测试集数据：我们的面试题目标是预测 1 到 100的fizz buzz情况. 所以为了
# 更加公平的预测，不让分类预测器较早的知道要预测的数据的情况，
# 我们选取101到200这个范围的数作为我们的训练集和测试集。
# Note: 语法说明。 for i in range(101, 200)代表Python中从for循环中遍历取值为i，并
# 赋值将i值输入到feature_engineer函数
#训练集真题
x_train = np.array([feature_engineer(i) for i in range(101, 200)],dtype=float)
preprocessing.scale(x_train)
y_train = np.array([construct_sample_label(i) for i in range(101, 200)])

#plt.scatter(x_train[:,0],y_train[:])
#plt.show()


# #测试集期末考试试卷
x_test = np.array([feature_engineer(i) for i in range(1, 100)],dtype=np.float)
preprocessing.scale(x_test)
y_test = np.array([construct_sample_label(i) for i in range(1, 100)])





# # 构造线性回归模型 Y = AX。这里的X是代表含有三个维度的数组[x1,x2,x3]，
#regr = linear_model.LinearRegression()
regr = SVC()

# # 使用训练集训练模型，训练的过程，就是搜索最优A的值的过程。最优咱们通过
# # 均方差 = 根号下 (累加所有样本（真实值 - 预测值）^2) 来评价模型中A值选取的优劣
regr.fit(x_train, y_train)

# # 使用测试集进行预测
y_pred = regr.predict(x_test)
print("test true result")
print(y_test[0:5])
print("test predict result")
print(y_pred[0:5])
# # 使用均方差（mean squared error）评价模型，均方差越小，
# # 代表模型越精准
print("Mean squared error : %.2f"
      % mean_squared_error(y_test, y_pred))

print('input number %s' %sys.argv[1])

i_featrue = np.array([feature_engineer(float(sys.argv[1]))])

preprocessing.scale(i_featrue)
result_data =regr.predict(i_featrue)
print('predict value %s' %(output(result_data,sys.argv[1])))
#feature argv
#print(regr.coef_)
#feature diff
#print(regr.intercept_)
#model argv
#print(regr.get_params)
#model score
print(regr.score(x_train,y_train))#use example 2^2
