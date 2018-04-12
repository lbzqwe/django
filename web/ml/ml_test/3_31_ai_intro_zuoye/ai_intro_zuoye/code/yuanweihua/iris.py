from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 
iris = load_iris()
print(iris.data[0:100]) # X
print(iris.target[0:100]) # Y 
X = iris.data
y = iris.target
# 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

knn = KNeighborsClassifier(n_neighbors=5)

# (2) Model training
knn.fit(X_train, y_train)

# (3) Predict & Estimate the score
y_pred = knn.predict(X_test)
print("y_test")
print(y_test[0:5])
print("y_pred")
print(y_pred[0:5])
print(knn.score(X_test, y_test))

# (5) Parameter tuining:
# this is how to use cross_val_score to choose model and configs #
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
k_range = range(1, 10)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') # for classification
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
