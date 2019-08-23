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
import matplotlib.pyplot as plt
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

#%% p60
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, .1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color = 'k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y axis ticks and gridline
plt.yticks([0.0, 0.5,  1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.show()

#%% p64
def cost_1(z):
    return -np.log(sigmoid(z))
def cost_0(z):
    return -np.log(1-sigmoid(z))
z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)
cl = [cost_1(x) for x in z]
plt.plot(phi_z, cl, label = 'J(w) if y = 1')
c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle = '--', label = 'J(w) if y = 0')
plt.ylim(0.0, 5.1)
plt.xlim([0,1])
plt.show()

#%% p68
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta = 0.05, n_iter = 1000, random_state = 1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X = X_train_01_subset, y = y_train_01_subset,
                      classifier = lrgd)

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()

#%% p71
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 100.0, random_state = 1)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, 
                      classifier = lr, test_idx = range(105, 150))

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()

lr.predict_proba(X_test_std[:3, :])

lr.predict_proba(X_test_std[:3, :]).sum(axis = 1)
lr.predict_proba(X_test_std[:3, :]).argmax(axis = 1)
lr.predict(X_test_std[:3, :])
lr.predict(X_test_std[0, :].reshape(1,-1))

#%% p 74 read one more time on ipad
#%% p75
weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C = 10.**c, random_state = 1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
    
weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label = 'petal length')
plt.plot(params, weights[:, 1], linestyle = '--',
         label = 'petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc = 'upper left')
plt.xscale('log')
plt.show()

#%% p80
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C = 100.0, random_state = 1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined, 
                      classifier = svm,
                      test_idx = range(105, 150))

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()

#%% pg 81 use these on large amounts of data. 
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss = 'perceptron')
lr = SGDClassifier(loss = 'log')
svm = SGDClassifier(loss = 'hinge')

#%% p82
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, 
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], 
            X_xor[y_xor == 1, 1],
            c = 'b', marker = 'x', label = '1')
plt.scatter(X_xor[y_xor == -1, 0], 
            X_xor[y_xor == -1, 1],
            c = 'r', marker = 's', label = '-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc = 'best')
plt.show()

#%% p85
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', random_state = 1, gamma = 0.1, C = 10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier = svm)
plt.legend(loc = 'upper left')
plt.show()

#%% p86
svm = SVC(kernel = 'rbf', random_state = 1, gamma = 0.2, C = 1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, 
                      classifier = svm, test_idx = range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()
#%% p87
svm = SVC(kernel = 'rbf', random_state = 1, gamma = 100.0, C = 1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, 
                      classifier = svm, test_idx = range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()

#%% p94
import matplotlib.pyplot as plt
import numpy as np
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







