<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 00:39:10 2019

@author: 15306
"""

from math import pi
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:17:35 2019

@author: stevxu
"""

import numpy as np

from numpy import random as rand

from math import pi

def approx_pi(n_samp):
    """
    Approximate pi using monte carlo methods by generating n_samp bivariate 
    uniform random points in [-1,1] x [-1,1]. 
    """
    n_samp = np.int(n_samp)
    
    x = rand.uniform(-1,1,n_samp)
    
    y = rand.uniform(-1,1,n_samp)
    
    z = (x**2+y**2)<1
    
    est_pi = np.sum(z) / n_samp * 4
    
    return est_pi


"""
Approximation with 1e2
"""
np.abs(pi - approx_pi(1e2))

"""
Approximation with 1e3
"""
np.abs(pi - approx_pi(1e3))

"""
Approximation with 1e5
"""
np.abs(pi - approx_pi(1e5))

"""
Approximation with 1e6
"""
np.abs(pi - approx_pi(1e6))
>>>>>>> 5e9330399ac2067d03b20d2634316aee265f9196
