# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:33:43 2023

@author: patri
"""
import numpy as np
from scipy.stats import norm

##### IMplied Volatilities modules #####
##### Bachelier, Black model ####

class ImVol(): 
    def __init__():
        return 0
    
def Black( F , K , V , w = 1):
        _d1 = w * (np.log(F / K) / V + 0.5 * V)
        _d2 = _d1 - V
        a = w * F * norm.cdf(_d1)
        b = - w * K * norm.cdf(_d2)
        return a + b
    
def Bachelier( F , K , V , w = 1 ):
    d = ( F - K ) / V
    a = (F-K) * norm.cdf(d)
    b = V * norm.pdf(d)
    return a + b