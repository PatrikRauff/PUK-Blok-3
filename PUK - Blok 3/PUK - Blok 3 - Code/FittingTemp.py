# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 18:16:38 2023

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
import SpreadModel as SM
######################################################
################# Test Enviroment ####################
######################################################

## Model specific - Tenor Structure ##
Tenor = 0.25
tnStr = '3M'
if Tenor == 0.5:
    tnStr = '6M'

## Short Rate parameters ##
## Gaussian ##
srVol = 0.011
MeanReversionRate = -0.06
meanLevel = .0253
r0 = 0.0279

## Spread Basis Parameters ##
## Log Normal model ##
eta = 0.50

## Correlation ##
rho = 0. ### Spread independent of RFR FORWARD ###
FwdDep = False
if FwdDep:
    rho = (Tenor == 0.25) * -0.6023 + \
        (Tenor == 0.5) * -0.6210
   
print("Correlation Param: ", rho)

## init xIBOR ##
## EURIBOR and EUROSWAP ##
## Daycount 30/360
# Tenors of 1W  , 1M , 3M , 6M , 12M Euribor

TimePoints = np.array([ 0.0194444 , 0.0388889 	, 
                       0.50 	, 0.75 ,	 1.00 ,	 
                       2.00 ,	 3.00 ,	 4.00 	, 5.00 ,	
                       6.00 ,	 7.00 ,	 8.00 	, 9.00 ,	 
                       10.00, 	 11.00 ,	 12.00 	, 15.00 	
                       , 17.00 	, 20.00 ,	 25.00 	, 30.00  ] )

### Default is 3M EUR Swap Curve
obsPoints = np.array([ 2.7967000 	, 2.8305000 ,3.054,	
                      3.1605,	3.1956	,3.12565,	
                      3.0175	,2.928	,2.8779	,2.84335	
                      ,2.82795,	2.82245	,2.83	,2.841	
                      ,2.8579	,2.87475	,2.8794,	
                      2.838,	2.7302,	2.53775	,2.3672 ] )/100
if Tenor == 0.5: ###Basis swapped EUR Swap curve for 6M 
    obsPoints = np.array([ 2.797     ,	 2.831  ,   	 
                          3.078     ,	 3.209  ,   	 
                          3.260     ,	 3.195  ,   	 
                          3.085     ,	 2.992  ,   	 
                          2.933     ,	 2.890  ,   	 
                          2.865     ,	 2.849  ,   	 
                          2.846     ,	 2.847  ,   	 
                          2.853     ,	 2.858  ,   	 
                          2.838     ,	 2.783  ,   	 
                          2.666     ,	 2.461  ,   	
                          2.282     ])/100


######### PLaying With Nelson Siegel Fitting on  ##############
# NS = Fitter.NelsonSiegelFitting(TimePoints, obsPoints , 'NS')
NSS = Fitter.NelsonSiegelFitting(TimePoints, obsPoints , 'NSS')
# # NSFit = NS.yFit(TimePoints)
NSSFit = NSS.yFit(TimePoints)

TauArr = np.arange(0.01 , 30.01 , 0.01)
plt.figure()
plt.title("Nelson-Siegel-Svensson Fitted EUR SWAP - Tenor: " + tnStr)
plt.plot(TimePoints , obsPoints , label = "Obs EUR SWAP")
plt.scatter(TimePoints , obsPoints , label = "Obs EUR Points" , marker = 'x' , s = 20)
plt.scatter(TimePoints , NSSFit , label = "NSS Fit EUR Curve" , marker = 'x' , s = 20)
plt.plot(TauArr , NSS.yFit(TauArr) , label = "NSS")
plt.xlabel("Time")
plt.ylabel("ZCB Yield")
plt.legend()

# NSzcb = NS.yFitZcb(TimePoints)
NSSzcb = NSS.yFitZcb(TimePoints)
plt.figure()
# plt.scatter(TimePoints , NSzcb , label = "NS Fit ZCB Curve")
# plt.plot(TauArr , NS.yFitZcb(TauArr) , label = "NS")
plt.scatter(TimePoints , NSSzcb , s = 20 , marker = 'x' , label = "Fitted NSS ZCB ")
plt.plot(TauArr , NSS.yFitZcb(TauArr) , label = "Fitted NSS Curve")
plt.legend()

E = TauArr + Tenor
plt.figure()
fwdNSS = (NSS.yFitZcb(TauArr)/NSS.yFitZcb(E)-1)/Tenor
# fwdNS = (NS.yFitZcb(TauArr)/NS.yFitZcb(E)-1)/Tenor
plt.plot(TauArr ,fwdNSS , label = tnStr + ' EURIBOR'  )
# plt.plot(TauArr ,fwdNS , label = 'Fwd NS ' )
plt.title("Fitted Forward Rates Tenor: " + tnStr )
plt.xlabel("Time")
plt.legend()