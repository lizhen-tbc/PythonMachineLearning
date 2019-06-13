# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 06:39:44 2019

@author: lizhe
"""

class AdalineSGD(object):
    """ Adaptive Linear Neuron classifier. 
    
    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle: bool (default: True)
        shuffles training data every epoch if True to prevent cycles
    random_state : int
        Random number generator seed for random weight initialization
        
    Attributes
    -------------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch
    
    """