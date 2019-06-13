# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 07:30:03 2019

@author: Zhen.Li
"""

# import AdalinLinearNeronCh2

fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (4, 5))

ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')

ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')

ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter = 10, eta = 0.0001). fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker = 'o')

ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')

ax[1].set_title('Adaline - Learning rate 0.0001')


ada3 = AdalineGD(n_iter = 10, eta = 0.001). fit(X, y)
ax[2].plot(range(1, len(ada3.cost_) + 1), np.log10(ada3.cost_), marker = 'o')

ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('log(Sum-squared-error)')

ax[2].set_title('Adaline - Learning rate 0.001')

plt.show()

#%%
X_std = np.copy(X)
X_std[:,0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:,1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
#%%
# use Adaline SGD
ada = AdalineSGD(n_iter = 15, eta = 0.01, random_state = 1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - stochastic gradient descent')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('average cost')
plt.show()


















 
