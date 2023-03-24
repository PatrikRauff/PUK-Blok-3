# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:30:40 2023

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
FwdDep = True
if FwdDep:
    rho = (Tenor == 0.25) * -0.6023 + \
        (Tenor == 0.5) * -0.6210
   
print("Correlation Param: ", rho)

## Spread Vol ##
v_0 = 0.
v_1 = 0.
v_2 = 0.
v_3 = 0.

########### MOdelling ############

Dep_structures= np.array([ 
    [0.0 , 0.0] 
    ,[0. , 0.0 ] 
    ,[0.0 , 0.0] 
    , [0.0 , 0.0] 
    ])

if Tenor == 0.25:
    v_1 = 0.0218 / 2
    v_2 = 0.0332 / 2
    v_3 = 0.0449 / 2
    a1 = 2 * v_1 * rho
    b1 = np.sqrt( 1- np.square(rho)) * v_1
    a2 = 2* v_2 * rho
    b2 = np.sqrt( 1- np.square(rho)) * v_2
    a3 = 2* v_3 * rho
    b3 = np.sqrt( 1- np.square(rho)) * v_3
    Dep_structures[ 1 , 0] = np.round(a1 , 5)
    Dep_structures[ 1 , 1] = np.round(b1 , 5)
    Dep_structures[ 2 , 0] = np.round(a2 , 5)
    Dep_structures[ 2 , 1] = np.round(b2 , 5)
    Dep_structures[ 3 , 0] = np.round(a3 , 5)
    Dep_structures[ 3 , 1] = np.round(b3 , 5)
       
if Tenor == 0.5:
    v_1 = 0.0245 * 0.5
    v_2 = 0.0287 * 0.5
    v_3 = 0.0331 * 0.5
    a1 = 2* v_1 * rho
    b1 = np.sqrt( 1- np.square(rho)) * v_1
    a2 = 2*v_2 * rho
    b2 = np.sqrt( 1- np.square(rho)) * v_2
    a3 = 2*v_3 * rho 
    b3 = np.sqrt( 1- np.square(rho)) * v_3
    Dep_structures[ 1 , 0] = np.round(a1 , 5)
    Dep_structures[ 1 , 1] = np.round(b1 , 5)
    Dep_structures[ 2 , 0] = np.round(a2 , 5)
    Dep_structures[ 2 , 1] = np.round(b2 , 5)
    Dep_structures[ 3 , 0] = np.round(a3 , 5)
    Dep_structures[ 3 , 1] = np.round(b3 , 5)

print(Dep_structures)
## Initiate Spread Model ##
Model = SM.SpreadModel(srVol, MeanReversionRate, meanLevel, r0, eta, Tenor)

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



### Fitting initial Libor Curve ###
Model._SetInitialLiborCurve(TimePoints, obsPoints)


#### Initiate OIS Model #####

OISModel = SR.Vasicek(srVol, MeanReversionRate, meanLevel, r0)

## Try pricing a Caplet ##
# With Strike 0.03 = 3%
# Fixes in half a year
# Tenor is implied in the model - eg. 0.5

##### For Cap Pricing
Fixing = 0.25
Mat = 10. + Fixing
#K = 0.02

# ########## BLACK IMPLIED VOL ############
# # print("Initiate Black Implied VOLS")
# ### initial Data ###
# # ATM Strike ## ATM Implied VOL #
# Type = 'Black'
# print("Initiate" + Type + "Implied VOLS")
# atmK = Model._initSwapRate( Fixing, Mat )
# CapIV = Model.CapImVol(Fixing, atmK, Mat , model= Type)
# OISTenor = 1.
# OISAtm = OISModel.initSwapRate(Fixing, Mat, OISTenor)

# StrikeArr = np.arange(atmK - 0.0100, atmK + 0.01001 , 0.0025 ,  )


# ImvolArr = np.zeros(shape = (StrikeArr.shape[0] , Dep_structures.shape[0] ))

# for j in range(ImvolArr.shape[1]):
#     print("Dependence Structure")
#     print(Dep_structures[j,:])
#     Model._SpreadDependence(Dep_structures[j , 0] , Dep_structures[j, 1])
#     print("Recalibrating SR Vol to hit ATM IV for initial Model")
#     Model._CapVolCal(Fixing, Mat, atmK, CapIV , Type)
#     for i in range(ImvolArr.shape[0]):
#         # print("Iter: ",i+1)
#         #Price = Model.Cap(Fixing, StrikeArr[i] , Mat)
#         ImvolArr[i , j] = Model.CapImVol( Fixing, StrikeArr[i], Mat , model = Type)
# ###
# # OIS Prep ##
# OisStrikeArr = np.arange(OISAtm - 0.0100, OISAtm + 0.01001 , 0.0025 )
# OISIMVol = np.zeros(shape = StrikeArr.shape[0])

# ## Recalibrate OIS Model VOL ##
# OISModel._CapVolCal(OISTenor, Fixing, Mat, OISAtm, CapIV , Type)

# for i in range(OISIMVol.shape[0]):
#     OISIMVol[i] = OISModel.CapImVol(OISTenor, Fixing, Mat, OisStrikeArr[i] , Model = Type)
# plt.figure()
# for j in range(ImvolArr.shape[1]):
#     plt.plot((StrikeArr - atmK)*10000 , ImvolArr[:,j] , 
#           label = r'$\alpha$ =' + str(Dep_structures[j,0]) + r', $\beta$ =' + str(Dep_structures[j,1]) )
# # print(OISIMVol)
# ## Plot OIS ##
# plt.plot((OisStrikeArr - OISAtm)*10000 , OISIMVol , label= 'OIS Model')
# plt.legend()
# plt.title(r'Model: $(S , E , \tau , \sigma , \beta , \eta) = ($' 
#           + str(Fixing) + r',' + str(int(Mat-Fixing)) + r'Y,' + tnStr + r','  
#           + str(srVol) + r',' + str(MeanReversionRate) + r',' + str(eta) + r')' 
#           )
# plt.xlabel("Strike")
# plt.ylabel(Type + r' Im. Vol.')

# ########## BLACK IMPLIED VOL #############




# # ############ BACHELIER IMPLIED VOL ##########
# print("########## INITIATE IMPLIED BACHELIER MODEL ##########")
# ## initial Data  - Refresh Class ###
# del Model
# del OISModel
# Model = SM.SpreadModel(srVol, MeanReversionRate, meanLevel, r0, eta, Tenor)
# OISModel = SR.Vasicek(srVol, MeanReversionRate, meanLevel, r0)
# ### Fitting initial Libor Curve ###
# Model._SetInitialLiborCurve(TimePoints, obsPoints)

# Type = 'Bachelier'
# print("Initiate" + Type + "Implied VOLS")
# atmK = Model._initSwapRate( Fixing, Mat )
# CapIV = Model.CapImVol(Fixing, atmK, Mat , model= Type)
# OISTenor = 1.
# OISAtm = OISModel.initSwapRate(Fixing, Mat, OISTenor)

# StrikeArr = np.arange(atmK - 0.0100, atmK + 0.01001 , 0.0025 ,  )


# ImvolArr = np.zeros(shape = (StrikeArr.shape[0] , Dep_structures.shape[0] ))

# for j in range(ImvolArr.shape[1]):
#     print("Dependence Structure")
#     print(Dep_structures[j,:])
#     Model._SpreadDependence(Dep_structures[j , 0] , Dep_structures[j, 1])
#     print("Recalibrating SR Vol to hit ATM IV for initial Model")
#     Model._CapVolCal(Fixing, Mat, atmK, CapIV , Type)
#     for i in range(ImvolArr.shape[0]):
#         # print("Iter: ",i+1)
#         #Price = Model.Cap(Fixing, StrikeArr[i] , Mat)
#         ImvolArr[i , j] = Model.CapImVol( Fixing, StrikeArr[i], Mat , model = Type)
# ###
# ## OIS Prep ##
# OisStrikeArr = np.arange(OISAtm - 0.0100, OISAtm + 0.01001 , 0.0025 )
# OISIMVol = np.zeros(shape = StrikeArr.shape[0])

# ## Recalibrate OIS Model VOL ##
# OISModel._CapVolCal(OISTenor, Fixing, Mat, OISAtm, CapIV , Type)

# for i in range(OISIMVol.shape[0]):
#     OISIMVol[i] = OISModel.CapImVol(OISTenor, Fixing, Mat, OisStrikeArr[i] , Model = Type)
# plt.figure()
# for j in range(ImvolArr.shape[1]):
#     plt.plot((StrikeArr - atmK)*10000 , ImvolArr[:,j] , 
#           label = r'$\alpha$ =' + str(Dep_structures[j,0]) + r', $\beta$ =' + str(Dep_structures[j,1]) )
# # print(OISIMVol)
# ## Plot OIS ##
# plt.plot((OisStrikeArr - OISAtm)*10000 , OISIMVol , label= 'OIS Model')
# plt.legend()
# plt.title(r'Model: $(S , E , \tau , \sigma , \beta , \eta) = ($' 
#           + str(Fixing) + r',' + str(int(Mat-Fixing)) + r'Y,' + tnStr + r','  
#           + str(srVol) + r',' + str(MeanReversionRate) + r',' + str(eta) + r')' 
#           )
# plt.xlabel("Strike")
# plt.ylabel(Type + r' Im. Vol.')
# ############ BACHELIER IMPLIED VOL ##########



########## Pricing Of CAPS in OIS ###############
Type = 'Black'
print("Initiate CAP Price calculation")
atmK = Model._initSwapRate( Fixing, Mat )
CapIV = Model.CapImVol(Fixing, atmK, Mat , model= Type)
OISTenor = 1.
OISAtm = OISModel.initSwapRate(Fixing, Mat, OISTenor)

Notional = 100

StrikeArr = np.arange(atmK - 0.0100, atmK + 0.01001 , 0.0025 ,  )
TenorArr = np.arange(5.0 , 25.01 , 5.) + Fixing

ImvolArr = np.zeros(shape = (StrikeArr.shape[0] , Dep_structures.shape[0] ))
for i in range(TenorArr.shape[0]):
    atmK = Model._initSwapRate( Fixing, TenorArr[i] ) + 0.005 
    
    #### JUST ATM ####
    for j in range(ImvolArr.shape[1]):
        # print("Dependence Structure: " , Dep_structures[j,:])
        Model._SpreadDependence(Dep_structures[j , 0] , Dep_structures[j, 1])
        print("Dep: " , Dep_structures[j,:] ," Cap for ", tnStr , " and Mat" , str(int(TenorArr[i]-Fixing)), "Y" , 
                  " | Strike: " , np.round(atmK*100 , 2) , " --- ",np.round( Notional*Model.Cap(Fixing, atmK , Maturity = TenorArr[i]) , 2) , "%" )
    print("OIS Price for ", tnStr , " and Mat" , str(int(TenorArr[i]-Fixing)), "Y" , 
              " | Strike: " , np.round(atmK*100 , 2) , " --- " , np.round(Notional * OISModel.Cap(r0, OISTenor, Fixing, TenorArr[i], atmK) ,2) , "%")
    
    # for k in range(StrikeArr.shape[0]):
    ## First Without Calibrating Vol ##
    # print("Recalibrating SR Vol to hit ATM IV for initial Model")
        # Model._CapVolCal(Fixing, TenorArr[i], atmK, CapIV , Type)
    # StrikeArr = np.arange(atmK - 0.0100, atmK + 0.01001 , 0.0025 ,  )
    # for k in range(StrikeArr.shape[0]):
    #     for j in range(ImvolArr.shape[1]):
    #         # print("Dependence Structure: " , Dep_structures[j,:])
    #         Model._SpreadDependence(Dep_structures[j , 0] , Dep_structures[j, 1])
    #         print("Dependence Structure: " , Dep_structures[j,:] ," Cap for ", tnStr , " and Mat" , str(int(TenorArr[i]-Fixing)), "Y" , 
    #                   " | Strike: " , StrikeArr[k] , " --- ",np.round( Notional*Model.Cap(Fixing, StrikeArr[k] , Maturity = TenorArr[i]) , 2) , "%" )
    #     print("OIS Price for ", tnStr , " and Maturity" , str(int(TenorArr[i]-Fixing)), "Y" , 
    #               " | Strike: " , StrikeArr[k] , " --- " , np.round(Notional * OISModel.Cap(r0, OISTenor, Fixing, TenorArr[i], StrikeArr[k]) ,2) , "%")
        

























##################### OLD BUT MAYBE USEFUL ##########################
# Type = 'Black'
# atmK = Model._initSwapRate( Fixing, Mat )
# print("Spread Model Atm Strike: ", atmK)
# CapIV = Model.CapImVol(Fixing, atmK, Mat , model= Type)
# OISAtm = Model._srModel.initSwapRate(Fixing, Mat, Tenor)
# print("OIS atm Strike: ", OISAtm)
# print("At the money Swap rate: ",atmK )
# print("Price of Caplet in spread model:")
# print(Model.Caplet(Fixing, atmK))
# print("Price of caplet in OIS model: ")
# print(OISModel.Caplet(r0, Fixing , Tenor, atmK))
# print("Price of Cap in Spread Model: ")
# print(Model.Cap(Fixing, atmK , Maturity= Mat))
# print("Price of Cap in OIS model: ")
# print(OISModel.Cap(r0, Tenor ,  Fixing  , Mat , atmK))

# Dep_structures= np.array([ 
#     [0.0 , 0.0] 
#     # ,[-0.01 , 0.003]
#     ,[-0.1 , 0.01 ] 
#     ,[0.05 , 0.015] 
#     , [0.07 , 0.02] 
#     ])


#### FOR MODEL CAPLETS ####
# cplATM = Model._initFwdLib(Fixing , Fixing + Tenor)
# TargetIV = Model.CapletImVol(Fixing, cplATM , model = Type)
# print('Caplet ATM (Fwd LIBOR): ' , cplATM)
# CplArr = np.arange(cplATM - 0.0100 , cplATM + +0.0101 , 0.0025)
# imCplVol = np.zeros(shape = (CplArr.shape[0] , Dep_structures.shape[0] ))
# for j in range(imCplVol.shape[1]):
#     print("Dependence Structure")
#     print(Dep_structures[j,:])
#     Model._SpreadDependence(Dep_structures[j , 0] , Dep_structures[j, 1])
#     print("Recalibrates SR VOL")
#     Model._cplVolCal(Fixing, cplATM, TargetIV)
#     for i in range(imCplVol.shape[0]):
#         imCplVol[i , j] = Model.CapletImVol(Fixing, CplArr[i] , model = Type)
# plt.figure()
# for j in range(imCplVol.shape[1]):
#     plt.plot((CplArr - cplATM)*10000 , imCplVol[:,j] , 
#           label = r'$\alpha$ =' + str(Dep_structures[j,0]) + r', $\beta$ =' + str(Dep_structures[j,1]) )
# plt.legend()
# plt.title(r'Model: $(S , \tau , \sigma , \beta , \eta) = ($' + str(Fixing) + r',' + str(Tenor) + r','  + str(srVol) + r',' + str(MeanReversionRate) + r',' + str(eta) + r')')
# plt.xlabel("Strike")
# plt.ylabel(Type + r' Cpl. Im. Vol.')  





# ######### PLaying With Nelson Siegel Fitting on  ##############
# # NS = Fitter.NelsonSiegelFitting(TimePoints, obsPoints , 'NS')
# NSS = Fitter.NelsonSiegelFitting(TimePoints, obsPoints , 'NSS')
# # # NSFit = NS.yFit(TimePoints)
# NSSFit = NSS.yFit(TimePoints)

# TauArr = np.arange(0.01 , 30.01 , 0.01)
# plt.figure()
# plt.title("Nelson-Siegel-Svensson Fitted EUR SWAP - Tenor: " + tnStr)
# plt.plot(TimePoints , obsPoints , label = "Obs EUR SWAP")
# plt.scatter(TimePoints , obsPoints , label = "Obs EUR Points" , marker = 'x' , s = 20)
# plt.scatter(TimePoints , NSSFit , label = "NSS Fit EUR Curve" , marker = 'x' , s = 20)
# plt.plot(TauArr , NSS.yFit(TauArr) , label = "NSS")
# plt.xlabel("Time")
# plt.ylabel("ZCB Yield")
# plt.legend()

# # NSzcb = NS.yFitZcb(TimePoints)
# NSSzcb = NSS.yFitZcb(TimePoints)
# plt.figure()
# # plt.scatter(TimePoints , NSzcb , label = "NS Fit ZCB Curve")
# # plt.plot(TauArr , NS.yFitZcb(TauArr) , label = "NS")
# plt.scatter(TimePoints , NSSzcb , s = 20 , marker = 'x' , label = "Fitted NSS ZCB ")
# plt.plot(TauArr , NSS.yFitZcb(TauArr) , label = "Fitted NSS Curve")
# plt.legend()

# Tenor = 0.5
# E = TauArr + Tenor
# plt.figure()
# fwdNSS = (NSS.yFitZcb(TauArr)/NSS.yFitZcb(E)-1)/Tenor
# # fwdNS = (NS.yFitZcb(TauArr)/NS.yFitZcb(E)-1)/Tenor
# plt.plot(TauArr ,fwdNSS , label = 'Fwd NSS'  )
# # plt.plot(TauArr ,fwdNS , label = 'Fwd NS ' )
# plt.title("Fitted Forward Rates Tenor: " + str(Tenor))
# plt.legend()




# EUR Swap 2Y , 3Y  , 4Y , 5Y , 6Y , 7Y , 8Y , 9Y , 10Y , 12Y , 15Y , 20Y , 30Y
# TimePoints = np.array([7/360 , 30/360 , 90/360 , 180/360 ,
#                    1. , 2. , 3. , 4. , 5. , 6. , 7. ,
#                    8. , 9. , 10. , 12. , 15. ])
#                    # 20. ,25. , 30.])

# obsPoints = np.array([2.482 , 2.648 , 2.750 , 3.055 ,
#                    3.3380 , 3.2030 , 3.0970 , 2.9810 , 2.9260 , 2.8860 , 2.8620 , 
#                    2.8480 , 2.8460 , 2.8490 , 2.8610 , 2.8480  ])/100
#                    # , 2.6750 ,2.4650,  2.2970]



