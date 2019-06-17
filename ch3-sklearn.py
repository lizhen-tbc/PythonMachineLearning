# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 07:34:28 2019

@author: Zhen.Li
"""
# p53
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('class labels:', np.unique(y))

#%% p54
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 1, stratify = y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

#%% p54
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#%% p55
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassification samples: %d' % (y_test != y_pred).sum())

#%%
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

#%%
y_pred1 = np.copy(y_pred)
y_pred1 = ["%.2f" % i for i in y_pred]
y_pred1 = np.array(y_pred1, dtype = '<U6')
y_pred1[y_pred1 == "0.00"] = "red"
y_pred1[y_pred1 == "1.00"] = "blue"
y_pred1[y_pred1 == "2.00"] = "yellow"

plt.scatter(X_test[:, 0], X_test[:, 1],
            color = y_pred1, marker = 'o')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()

#%%
y_test1 = np.copy(y_test)
y_test1 = ["%.2f" % i for i in y_test]
y_test1 = np.array(y_test1, dtype = '<U6')
y_test1[y_test1 == "0.00"] = "red"
y_test1[y_test1 == "1.00"] = "blue"
y_test1[y_test1 == "2.00"] = "yellow"

plt.scatter(X_test[:, 0], X_test[:, 1],
            color = y_test1, marker = 'o')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()
#%% p57
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X = X_combined_std,
                      y = y_combined,
                      classifier = ppn, 
                      test_idx = range(105, 150))
plt.xlabel('petal length ')
plt.ylabel('petal width ')
plt.legend(loc = 'upper left')
plt.show()




