# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:58:15 2023

@author: patri
"""
import numpy as np
from scipy.optimize import curve_fit, minimize, root

### Nelson Siegel efitting module ###
class NelsonSiegelFitting():
    def __init__(self , Tenors : np.array , points : np.array , Model = 'NS' ):
        self._TimeToMat = Tenors 
        self._YieldPoints = points
        self._model = Model
        self.fitModel()
    
    def yNSS(self , tau ,  b0 , b1 , b2 , b3 , l1 , l2 ):
        tau = np.where(tau == 0 , 0.001 , tau)
        t1 = ( 1 - np.exp(- l1 * tau) ) / (l1 * tau)
        t2 = t1 - np.exp( - l1 * tau)
        t3 = (1 - np.exp( - tau * l2)) / (l2 * tau) - np.exp(- l2 * tau)
        return b0 + b1 * t1 + b2 *t2 + b3 * t3
    
    def yNS(self , tau ,  b0 , b1 , b2 , lamb ):
        tau = np.where(tau == 0 , 0.001 , tau)
        t1 = lamb *( 1 - np.exp(- lamb * tau) ) / (lamb * tau)
        t2 = t1 - np.exp( - lamb * tau)
        return b0 + b1 * t1 + b2 *t2
    
    def yFit(self , tau):
        if self._model == 'NS':
            b0 = self.b0
            b1 = self.b1
            b2 = self.b2
            lamb = self.lamb
            return self.yNS(tau, b0, b1, b2, lamb)
        
        if self._model == 'NSS':
            b0 = self.b0
            b1 = self.b1
            b2 = self.b2
            b3 = self.b3
            l1 = self.l1
            l2 = self.l2
            return self.yNSS(tau, b0, b1, b2, b3 , l1 , l2 )
    
    def yFitZcb(self , tau):
        yields = self.yFit(tau)
        return np.exp(- yields * tau )
    
    def fitModel(self):
        xobs = self._TimeToMat
        yobs = self._YieldPoints
        func = self.yNS
        ## Standard Nelson Siegel Fit ##
        if self._model == 'NS':
            FittedMod = curve_fit(func, xobs, yobs  )
            # print(FittedMod)
            resParams = FittedMod[0]
            self.b0 = resParams[0]
            self.b1 = resParams[1]
            self.b2 = resParams[2]
            self.lamb = resParams[3]
        ## Nelson Siegel Svensson model to fit ##
        if self._model == 'NSS':
            xobs = self._TimeToMat
            yobs = self._YieldPoints
            func = self.yNSS
            FittedMod = curve_fit(func, xobs, yobs  )
            # print(FittedMod)
            resParams = FittedMod[0]
            
            self.b0 = resParams[0]
            self.b1 = resParams[1]
            self.b2 = resParams[2]
            self.b3 = resParams[3]
            self.l1 = resParams[4]
            self.l2 = resParams[5]
            