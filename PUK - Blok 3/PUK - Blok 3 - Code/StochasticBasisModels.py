# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:17:24 2023

@author: patri
"""
import numpy as np

class Basis_model():
    def __init__(self , ShortVol , LongVol = 0. ,  Model = "LogNormal"):
        self.sVol = ShortVol
        self.lVol = LongVol
        self.distribution = Model
        self.iBasis = 1
       
    def _setiBasis(self , new_init):
        self._iBasis = new_init
    
    def _lnVariance(self , start , end):
       S = start
       E = end 
       _nVar = self._nVariance(S, E)
       return np.exp(_nVar) - 1
    
    def _nVariance(self, Start , end):
        S = Start
        E = end
        VolSqr = self.sVol*self.sVol
        return VolSqr * (E - S)
    