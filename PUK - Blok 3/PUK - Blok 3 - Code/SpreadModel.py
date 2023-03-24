# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:33:25 2023

@author: patri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import integrate
from scipy.optimize import curve_fit, root , newton , minimize

### Import own Modules ###
import ShortRateModels as SR
import StochasticBasisModels as BM    
import NelsonSiegel as Fitter  
import Helper as H 
    
class SpreadModel():
    def __init__(self , ShortRateVol , ShortRateMeanRev , 
                 ShortRateLevel , initRate, BasisVol , Tenor : float):
        print("Spread Class initiated with Tenor Basis of: " , str(Tenor))
        self._srVol = ShortRateVol
        self._bVol = BasisVol
        self._srBeta = ShortRateMeanRev
        self._srB  = ShortRateLevel
        self._Tenor = Tenor
        self._srModel = SR.Vasicek(ShortRateVol, ShortRateMeanRev, ShortRateLevel, initRate)
        self._bModel = BM.Basis_model(self._bVol)
        self._Spread = "Deterministic"
        self._a_k = 0.00
        self._b_k = 0.00
    
    def _setSRVol(self, vol ):
        self._srVol = vol
        self._srModel.setVol(vol)
    
        
    #Infer from Euro Swap curve from a given day?
    #Maybe fit Nelson Siegel to get continous curve?
    #Functions for initial Libor
    def _SetInitialLiborCurve(self , Tenors , Points , Method = 'NSS'):
        print("Fitting initial EUR Swap curve for Fwd Libor")
        self.LibFit = Fitter.NelsonSiegelFitting(Tenors, Points , Method)
    
    
    def _initFwdLib(self , S , E):
        delta = E - S
        _zcbS = self.LibFit.yFitZcb(S)
        _zcbE = self.LibFit.yFitZcb(E)
        Ratio = _zcbS / _zcbE
        if Ratio == 1:
            print("Warning: Initial Libor fwd rate is zero")
            return 0
        return (Ratio - 1) / delta
        
    def _SpreadDependence(self  , alpha = 0. , beta = 0.):
        self._b_k = beta
        self._a_k = alpha
    
   
    #### Make helper functions to end up having a function 
    ## for the semianalytical integral to compute numerically at last
    # what we need of paramaters:
    # Tenor, Strike , Start of contract 
    # What else is not determined initially in the model?
    
    ## Black Function
    def _Black(self, F , K , V , w = 1):
        _d1 = w * (np.log(F / K) / V + 0.5 * V)
        _d2 = _d1 - V
        a = w * F * norm.cdf(_d1)
        b = - w * K * norm.cdf(_d2)
        return a + b
    
    ## Option price of a bachelier option - TBD ##
    def _Bachelier(self , F , K , V , w = 1):
        return 0
    
    def _Helper(self , A , B , V):
        if A > 0 and B > 0:
            # print("both positive")
            return self._Black(A, B, V , 1)
        if A < 0 and B < 0:
            # print("Both Negative")
            return self._Black(-A, -B, V , -1)
        if A >= 0 and B <= 0:
            # print("A pos - B Neg")
            return A - B
        if A <= 0 and B >= 0:
            # print("A non.pos - B non-neg")
            return 0
    
    def _Integral(self , y , S , E , K ):
        ## Model implied parameters
        ## short rate moments
        # Mean
        # print(S , E)
        Ey = self._srModel._fwdMean(0, S )
        # print(Ey)
        # print(Ey)
        #Variance
        Vy = np.sqrt(self._srModel._fwdVariance(0, S))
        # print(Vy)
        # print(Vy)
        ## Spread Moments ##
        # Variance ==> STD
        V = np.sqrt( self._bModel._nVariance(0,S) )
        # V = np.sqrt( self._bModel._nVariance(S,E) )
        # print(V)
        ## Dependence structure
        ak = 1 + self._a_k
        bk = self._b_k
        
        ## Initially known
        iLib = self._initFwdLib(S , E)
        
        iFwd = self._srModel.InitFwdRate(S, E)
        xi = iLib - ak * iFwd - bk
        # print("xi_k: ", xi)
        # print(iLib)
        # print(iFwd)
        ## Given Tenor variables
        tau = E - S
        # print(y)
        #### State dependent
        Pk_y = self._srModel.ZeroCouponBond( S, E, y)
        # print(Pk_y)
        ### State Dependent Variables
        C_y = Pk_y * bk * tau
        D_y = Pk_y * (tau * K + ak - xi * tau) - ak
        # print("C_y: " , C_y , "\nD_y: ", D_y )
        ## H Function on statedependent variables ##
        H_y = self._Helper(C_y, D_y, V)
        # print(H_y)
        ## normal Density ##
        pdf_y = norm.pdf(y , loc = Ey , scale = Vy)
        # pdf_y = np.exp(-0.5 * np.square(y - Ey) / Vy ) / np.sqrt(2 * np.pi * Vy)
        
        # print(pdf_y)
        return H_y * pdf_y
    
    def _initSwapRate(self , Start , End):
        ### First fixing is start a
        S = Start
        E = End
        # print("start: ",S," | End: " , E)
        dt = self._Tenor
        tArr = np.arange(S, E , dt )
        ## first payment a+
        numIter = int( (E - S) / self._Tenor)
        libArr = np.zeros(shape = numIter)
        ZCBarr = np.zeros(shape = numIter) 
        for i in range(numIter):
            libArr[i] = self._initFwdLib(tArr[i] , tArr[i]+dt)
            ZCBarr[i] = self._srModel.InitZCB(tArr[i] + dt) * dt
        ### end is last payment
        # print(libArr)
        # print(ZCBarr)
        top = np.sum(libArr * ZCBarr)
        bot = np.sum(ZCBarr)
        return top / bot
    
    ### exogenous parameters.. 
    ## Tenor , strike , start
    ## Rest is model specific 
    def Caplet( self , FwdStart , Strike )-> float:
        Delta = self._Tenor
        S = FwdStart
        E = S + Delta
        K = Strike
        # print("start: ",S," | End: " , E)
       
        DF = self._srModel._ZCB(0, S) 
        # print(DF)
        integral = integrate.quad_vec(self._Integral , -np.inf , np.inf, args = (S , E , K) , epsrel=1e-12 )[0]
        #print(integral)
        price = DF * integral
        return price
    
    def Cap( self ,  FwdStart , Strike , Maturity = 0)-> float:
        temp = 0
        S = FwdStart
        E = Maturity
        K = Strike
        delta = self._Tenor
        if E == 0:
            E = S + self._Tenor
        numIter = int( (E - S) / self._Tenor)
        temp = 0
        for i in range(numIter):
            res = self.Caplet( S + i*delta, K )
            # print("Caplet: ",res)
            temp += res
        return temp
    
    def CapletImVol(self , FwdStart , Strike , model = 'Black'):
        _S = FwdStart
        _delta = self._Tenor
        _E = _S + _delta
        
        Price = self.Cap(_S, Strike )
        # print("Cpalet Price: ", Price)
        # _Tarr = np.arange(_S , _E , _delta)
        # _ZCBs = self._srModel.InitZCB(_Tarr + _delta)
        # print(_Tarr)
        # print(_ZCBs)
        func = self._Black
        if model == 'Bachelier':
            func = H.Bachelier
        def temp(Vol):
            temp = 0 
            V = Vol * np.sqrt(_S)
            # print(np.sqrt(_Tarr[i]))
            # print(v)
            _Lib = self._initFwdLib(_S , _E )
            # print(_Lib)
            zcb = self._srModel.InitZCB(_S + _delta)
            # print(zcb)
            # print("Iter num: ", i ," | Fixing start: ", _Tarr[i] ,  " | Initial Libor" , _Lib)
            temp += zcb * _delta * func(_Lib, Strike, V )   
            return temp - Price
        res =  root(temp , 0.2)
        imVol = res.x[0]
        return imVol
    
    def CapImVol(self , FwdStart , Strike , Maturity , model = 'Black'):
        _S = FwdStart
        _delta = self._Tenor
        _E = Maturity
        numIter = int( (_E - _S) / self._Tenor)
        
        Price = self.Cap(_S, Strike , _E)
        # print("Cpalet Price: ", Price)
        # _Tarr = np.arange(_S , _E , _delta)
        # _ZCBs = self._srModel.InitZCB(_Tarr + _delta)
        # print(_Tarr)
        # print(_ZCBs)
        func = self._Black
        if model == 'Bachelier':
            func = H.Bachelier
        def temp(Vol):
           temp = 0 
           for i in range(numIter):
               V = Vol * np.sqrt(_S + i*_delta)
               # print(np.sqrt(_Tarr[i]))
               # print(v)
               _Lib = self._initFwdLib(_S + i*_delta, _S + (1+i)*_delta)
               # print(_Lib)
               zcb = self._srModel.InitZCB(_S + (1+i)*_delta)
               # print(zcb)
               # print("Iter num: ", i ," | Fixing start: ", _Tarr[i] ,  " | Initial Libor" , _Lib)
               temp += zcb * _delta * func(_Lib, Strike, V )   
           return temp - Price
        res =  root(temp , 0.2 )
        imVol = res.x[0]
        return imVol
    
    ##### Recalibrates 
    def _cplVolCal(self , Start , atmK , imVol , model = 'Black'):
        S = Start
        K = atmK
        print("ImVol to hit: " , imVol)
        initVal = self._srVol
        def temp(srVol):
            print("New Short Rate vol: ", srVol)
            self._setSRVol(srVol)
            tempIV = self.CapletImVol(S, K , model)
            return tempIV - imVol
        res = root(temp , initVal )
        self._setSRVol(res.x[0])
        print(self._srVol)
        return 0
    
    def _CapVolCal(self , Start , Maturity , atmK , imVol , model = 'Black'):
        S = Start
        K = atmK
        E = Maturity
        # print("ImVol to hit: " , imVol)
        initVal = self._srVol
        def temp(srVol):
            # print("New Short Rate vol: ", srVol[0])
            self._setSRVol(srVol[0])
            tempIV = self.CapImVol( S , K , E , model)
            # print()
            return np.abs(tempIV - imVol)
        res = minimize( temp , initVal , method = 'Nelder-Mead' )
        # print(res.x[0])
        self._setSRVol(res.x[0])
        print("Calibrated SR Vol for Spread Model: ", self._srVol)
