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