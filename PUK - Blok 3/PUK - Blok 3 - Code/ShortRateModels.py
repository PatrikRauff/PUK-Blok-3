# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:16:20 2023

@author: patri
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import root , minimize

import Helper as H

class Vasicek():
    def __init__(self , sigma , ReversionRate , MeanLevel , InitRate):
        self.vol = sigma
        self.beta = ReversionRate
        self.b = MeanLevel * (- self.beta)
        self._initRate = InitRate
        
    def setVol(self , New_vol):
        self.vol = New_vol
    
    def setBeta(self , new_beta):
        self.beta = new_beta
        
        
    def _fwdVol(self , start , end):
        t = start
        T = end
        beta = self.beta
        temp =  np.exp( beta * ( T - t ) ) - 1
        return temp / beta
    
    def _fwdVariance(self ,  start , end , TermMeasure = 0):
        S = start
        E = end
        tau = E - S
        vol = self.vol
        beta = self.beta
        exp2b = np.exp(2* beta * tau) - 1     
        return 0.5 * vol * vol * exp2b / beta
        
    def _fwdMean(self , start , end , TermMeasure = 0):
        if TermMeasure == 0:
            TermEnd = end
        else:
            TermEnd = TermMeasure  
        S = start
        E = end
        tau = E - S
        vol = self.vol
        beta = self.beta
        k = -beta
        b = self.b
        theta = b / k
        B = (1- np.exp(- k * tau )) 
        r0 = self._initRate
        term1 = (theta - vol * vol / (k * k ) ) * B
        expb = np.exp(beta * (TermEnd - E) ) 
        exp2b = np.exp( beta * (TermEnd + E - 2 * S)) 
        term2 = 0.5 *  vol * vol *(expb - exp2b) / ( beta * beta )
        M = term1 + term2
        return r0 * np.exp(b* tau) + M
    
    def ZeroCouponBond(self ,  start , end , knownRate ):
        #Term measure means under which forward measure number in fraction of year, 0 is Risk-Neutral Measure
        t = start
        T = end
        tau = T - t
        beta = self.beta 
        k = -beta
        b = self.b
        theta = b / k
        vol = self.vol
        B = (np.exp(beta * tau ) - 1) / beta
        Rate = knownRate
        A = (theta - 0.5 * vol * vol / (k * k) )*(B - tau) \
            - 0.25 * vol * vol * B * B / k
        return np.exp( A - B * Rate)
    
    def _ZCB(self, start , end):
        return self.ZeroCouponBond(start, end, self._initRate)
    
    def InitZCB(self , end):
        return self._ZCB(0, end)
        
    def InitFwdCurve(self , StartPoint , Tenor, NumTenor):
        beta = self.beta
        b = self.b
        vol = self.vol
        TenArr = np.arange(start = 0. , 
                           stop = (NumTenor)*Tenor,
                           step = Tenor)
        temp1 = self._fwdVol(0, TenArr)
        temp2 = np.square(temp1)
        fwd = StartPoint * np.exp(beta * TenArr)
        return fwd + b * temp1 - 0.5 * vol * vol * temp2
    
    def InitFwdRate(self , Start , End):
        tau = End - Start
        p0 = self.InitZCB(Start)
        p1 = self.InitZCB(End)
        # print("Tenor: ", tau)
        ratio = p0 / p1
        # print("ZCB Ratio for fwd rate: ", ratio)
        return (ratio - 1) / tau
    
    def Caplet(self , InitRate , FwdStart , Tenor , Strike ):
        S = FwdStart
        E = S + Tenor
        delta = Tenor
        K = Strike
        kappa = 1 + delta * K
        _zcbS = self.InitZCB( S)
        #print("ZCB maturing at ",S,": ", _zcbS)
        _zcbE = self.InitZCB( E)
        #print("ZCB maturing at ",E,": ", _zcbE)
        _fwdVol = self.vol * np.sqrt ( 0.5 *  self._fwdVol(0, 2 * S) )
        _TenorVol = self._fwdVol(S, E)
        _sigma = _fwdVol * _TenorVol
        # print("Caplet Vol for:")
        # print("Start :" , FwdStart)
        # print("Tenor :" , Tenor)
        # print("Vol: ", _sigma)
        _d =  ( np.log(_zcbS ) - np.log(_zcbE * kappa) ) / _sigma + 0.5 * _sigma
        return _zcbS * norm.cdf(_d) - _zcbE * kappa * norm.cdf(_d - _sigma)
        
    def Cap(self , InitRate , Tenor , fwdStart , Maturity , Strike):
        numCaplets = int( (Maturity - fwdStart) / Tenor)
        res = 0
        for i in range(numCaplets):
            # print("iter: ", i)
            temp = self.Caplet(InitRate , fwdStart + i*Tenor , Tenor , Strike)
            # print(temp)
            res += temp
        return res
    
    def initSwapRate(self ,  start , End , Tenor):
        S = start
        E = End
        delta = Tenor
        P = np.arange(S + delta , E + delta, delta)
        # print(P)
        P0 = self.InitZCB(S)
        P = self.InitZCB(P)
        # print(P)
        pend = P[-1]
        # print(pend)
        A = np.sum(delta * P)
        # print(A)
        return (P0- pend) / A
    
    def CapImVol(self , Tenor , FwdStart , Maturity , Strike , Model = 'Black'):
        
        r0 = self._initRate 
        dt = Tenor
        S = FwdStart 
        E = Maturity
        K = Strike
        numIter = int( (E - S) / dt)
        func = H.Black
        if Model == 'Bachelier':
            func = H.Bachelier
        Price = self.Cap(r0 ,  dt , S , E , K)
        # print(Price)
        def temp(Vol):
           temp = 0 
           for i in range(numIter):
               v = Vol * np.sqrt(S + i*dt)
               F = self.InitFwdRate(S+i*dt, S + (1+i)*dt)
               ZCB = self.InitZCB(S + (1+i)*dt)
               # print("Iter num: ", i ," | Fixing start: ", _Tarr[i] ,  " | Initial Libor" , _Lib)
               temp += ZCB * dt * func(F, K, v) 
           # print(temp) 
           return temp - Price
        res =  root(temp , 1)
        # print(res)
        imVol = res.x[0]
        return imVol
    
    def _CapVolCal(self , Tenor , Start , Maturity , atmK , imVol , model = 'Black'):
        S = Start
        K = atmK
        E = Maturity
        print("ImVol to hit: " , imVol)
        initVal = self.vol
        def temp(srVol):
            # print("New Short Rate vol: ", srVol)
            self.setVol(srVol[0])
            tempIV = self.CapImVol( Tenor , S , E , K , model)
            return np.abs(tempIV - imVol)
        res = minimize( temp , initVal , method = 'Nelder-Mead' )
        # print(res.x[0])
        self.setVol(res.x[0])
        print("Calibrated Vol: " , self.vol)        

## Model specific - Tenor Structure ##
Tenor = 0.5


## Test enviroment
Vol = 0.011
MeanReversionRate = -0.06

meanLevel = .023 
r0 = 0.0279

## Tenor is Fraction of year: 0.25 is quarterly - 0.5 is Semi anually
Tenor = 0.5
Maturity = 10
NumPoints = int(Maturity / Tenor) + 1
Tenors = np.arange(0 , Maturity + Tenor  , Tenor)
kArr = np.arange(0.00 , 0.05 , 0.0001)
startArr = np.arange(1, 10 + 0.25, Tenor)
betaArr = np.arange(-1 , -0.00 , 0.01)
print("INitiate Model")
Model = Vasicek(Vol, MeanReversionRate, meanLevel , r0)

# plt.figure()
# plt.title( 'Caplet Price wrt to $K$')
# plt.plot(kArr , Model.Cap(r0, Tenor , 1 , 5 , kArr))

# print("############### Beta sensitivity #############")
# tempArr = np.zeros(shape = betaArr.shape[0] )
# for i in range(betaArr.shape[0]):
#     Model.setBeta(betaArr[i])
#     tempArr[i] = Model.Caplet(r0 , 0.5  , 1 , 0.020)
#     # print(tempArr[i])
# plt.figure()
# plt.title( r'Caplet wrt., $\beta$ - S=0.25 , $\delta=0.5$, $K=2\%$')
# plt.plot(betaArr ,tempArr)
# print("############### Beta sensitivity #############")

# print("############### Initial Forward Curve #################")
# Model.setBeta(MeanReversionRate)
# fwdCurve = Model.InitFwdCurve(r0, Tenor, NumPoints)
# ZCBCurve = np.zeros(NumPoints)
# for i in range(NumPoints):
#     ZCBCurve[i] = Model.InitZCB(Tenors[i])
 

# plt.figure()
# plt.title('Initial Forward Curve')
# plt.plot(Tenors , fwdCurve , label = "f(0,t)")
# plt.legend()
# plt.figure()
# plt.title('Time 0 ZCB Curve')
# plt.plot(Tenors , ZCBCurve , label = 'P(0,t)')
# plt.scatter(Tenors , ZCBCurve , marker = 'x' , c = 'r')
# plt.legend()
# #(ZCBCurve[20] / ZCBCurve[30] - 1) / (Tenors[30] - Tenors[20])
# print("################# INitial Fwd Curve ###############")

print("Fwd Rates - Tenor: " , Tenor)
fwdStart = np.arange(0, 15.01 , 0.01)
FwdEnd = fwdStart + Tenor
plt.figure() 
fwdRates = Model.InitFwdRate(fwdStart, FwdEnd)
plt.title('Initial Fwd Rates - Tenor:' + str(Tenor))
plt.plot(fwdStart , fwdRates , label = 'Vasicek Fwd Rates')
plt.legend()

Start = 0.05
Maturity =  10. + 0.5
ATMStrike = Model.initSwapRate(Start, Maturity, Tenor)
strike = 0.03
Type = 'Bachelier'
print("Price of Cap with: \n")
print("Start at: " , Start)
print("Maturues in: ",Maturity)
print("Tenor Structure of: ", 1/Tenor,"Payments a year")
print("Strike at: ", strike*100 ,"%")
print("Price: ", Model.Cap(r0, Tenor, Start, Maturity, ATMStrike) )

print("OIS Swap rate for " , Maturity , " year swap")
print(Model.initSwapRate(Start, Maturity, Tenor))
print("IMplied ATM Cap for: T = ", Maturity , 
      "Tenor ")
print(Model.CapImVol(Tenor, Start, Maturity, ATMStrike))

Type = 'Bachelier'
StrikeArr = np.arange(ATMStrike - 0.0100, ATMStrike + 0.0125 , 0.0025 )
ResArr = np.zeros(StrikeArr.shape[0])
for i in range(ResArr.shape[0]):
    ResArr[i] = Model.CapImVol(Tenor, Start, Maturity, StrikeArr[i] , Model = Type)
    
plt.figure()
plt.plot((StrikeArr - ATMStrike)*10000 , ResArr, label= "SR Model")
plt.legend()
plt.title(r"Bachelier Cap IM Vol of Vasicek Model - $T$="+ str(Maturity) )

Type = 'Black'
StrikeArr = np.arange(ATMStrike - 0.0100, ATMStrike + 0.0125 , 0.0025 )
ResArr = np.zeros(StrikeArr.shape[0])
for i in range(ResArr.shape[0]):
    ResArr[i] = Model.CapImVol(Tenor, Start, Maturity, StrikeArr[i] , Model = Type)
    
plt.figure()
plt.plot((StrikeArr - ATMStrike)*10000 , ResArr, label= "SR Model")
plt.legend()
plt.title(r"Black Cap IM Vol of Vasicek Model - $T$="+ str(Maturity) )


