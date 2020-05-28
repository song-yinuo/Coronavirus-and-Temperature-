# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:02:55 2020

@author: heton
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from scipy.stats import norm

data = pd.read_csv("D:/spring/434/proj/data final version.csv")
data = data.set_index(['Country'])
D = data[['Average Temperature-April','Average Temperature-May','Average Temperature-April&May']]
Y = data[['total_cases']]
Z =data[['Death rate','Population density (people per sq. km of land area)','GDP per capita (constant 2010 US$)','Current health expenditure(% of GDP)','Employment to population ratio, 15+, total (%) (modeled ILO estimate)','Unemployment, total (% of total labor force) (modeled ILO estimate)','Adjusted savings: education expenditure (% of GNI)','Urban population (% of total population)','People using at least basic sanitation services (% of population)','Physicians (per 1,000 people)','Haq index']]
X = np.concatenate((D,Z),axis=1)

#replace NA with mean
nan_model=Imputer(missing_values='NaN',strategy='mean',axis=1)
X0=nan_model.fit_transform(X)
#scale X : (X-mean)/std
muhat = np.mean(X0, axis = 0)
stdhat = np.std(X0, axis = 0)
Xtilde = (X0-muhat)/stdhat
#(Y-min)/min
minhat = np.min(Y, axis = 0)
Y0 = (Y-minhat)/minhat


D = Xtilde[:,0:3]
Z =Xtilde[:,3:15]


#1 Lasso as a whole, alpha=50
lambda0=50
lasso0=Lasso(alpha = lambda0)
lasso0.fit(Xtilde,Y0)
coef0 = lasso0.coef_
print(np.around(coef0,2))

#2 sample splitting to decide alpha

#(Y-min)/std
minhat = np.min(Y, axis = 0)
stdhatY = np.std(Y, axis = 0)
Y1 = (Y-minhat)/stdhatY

(Xt,Xv,Yt,Yv) = train_test_split(Xtilde,Y1,test_size = 0.25,random_state = 0)
 #calculate the fit for each value of lambda
 
Yv=Yv.to_numpy()
alpha_grid = np.geomspace(0.001,1)
res = np.zeros(alpha_grid.size)
for i in range(alpha_grid.size):
     lasso1 = Lasso(alpha=alpha_grid[i])
     lasso1.fit(Xt,Yt)
     Y_pred= lasso1.predict(Xv)
     res[i]=np.mean((Yv-Y_pred)**2)
plt.plot(res)
lambda1 = alpha_grid[38]


#3 add 2-degree interations
plo_int=PolynomialFeatures(degree=2,include_bias=False)
Xnew=plo_int.fit_transform(Xtilde)
muhat = np.mean(Xnew, axis = 0)
stdhat = np.std(Xnew, axis = 0)
Xn = (Xnew-muhat)/stdhat
(Xt,Xv,Yt,Yv) = train_test_split(Xn,Y1,test_size = 0.25,random_state = 0)
Yv=Yv.to_numpy()
alpha_grid = np.geomspace(0.001,1)
res = np.zeros(alpha_grid.size)
for i in range(alpha_grid.size):
     lasso2 = Lasso(alpha=alpha_grid[i],max_iter=1000)
     lasso2.fit(Xt,Yt)
     Y_pred= lasso2.predict(Xv)
     res[i]=np.mean((Yv-Y_pred)**2)
plt.plot(res)
lambda2 = alpha_grid[38]
#they are the same, no need to add high order interaction

lasso2 = Lasso(lambda2,max_iter=10000)
lasso2.fit(Xtilde,Y0)
coef2 = lasso2.coef_
print(np.around(coef2,4))

#killed 'Average Temperature-April&May',sample splitting select too many variables


#4. use BRT to find lambda
sigma = np.std(Y0)
(n,p)=Xtilde.shape
c=1.1
a=0.05
lambda3 = 2*c*sigma*norm.ppf(1-a/(2*p))/np.sqrt(n)
print(lambda3)
lasso3 = Lasso(lambda3.item(),max_iter=10000)
lasso3.fit(Xtilde,Y0)
coef3 = lasso3.coef_
print(np.around(coef3,4))
 
#BRT killed all variables

#Double lasso
 #1.Y~D,Z
# choose lambda0 = 50

lasso4 = Lasso(alpha=lambda0)
lasso4 =lasso4.fit(Xtilde,Y0)
coef4 = lasso4.coef_
print(np.around(coef4,4))
alphahat=coef4[0:3]
gamahat=coef4[3:15]

 #2.D~Z
 #(1)use BTR to choose lambda
sigma5 = np.std(D[:,0])
(n,p)=Z.shape
c=1.1
a=0.05
lambda5 = 2*c*sigma*norm.ppf(1-a/(2*p))/np.sqrt(n)
print(lambda5)
lasso5 = Lasso(alpha=lambda5.item())
lasso5 =lasso5.fit(Z,D[:,0])
coef5 = lasso5.coef_
print(np.around(coef5,4))
#killed all Z

 #(2)use cross validation to choose lambda
lasso5=LassoCV(cv=5)
lasso5 =lasso5.fit(Z,D[:,0])
coef5 = lasso5.coef_
print(np.around(coef5,4))
#killed all Z

 #(3)use sample splitting to choose lambda
(Zt,Zv,Dt,Dv) = train_test_split(Z,D,test_size = 0.25,random_state = 0)
#Dv=Dv.to_numpy()
alpha_grid = np.geomspace(0.001,1)
res5 = np.zeros(alpha_grid.size)
for i in range(alpha_grid.size):
     lasso5 = Lasso(alpha=alpha_grid[i],max_iter=1000)
     lasso5.fit(Z,D[:,0])
     D_pred= lasso5.predict(Zv)
     res5[i]=np.mean((Dv[:,0]-D_pred)**2)
plt.plot(res5)
lambda5 = alpha_grid[0]
lasso5 = Lasso(alpha=lambda5.item())
lasso5 =lasso5.fit(Z,D[:,0])
coef5 = lasso5.coef_
print(np.around(coef5,4))
#SELECT ALL Z

phihat= coef5
#plug in to estimate alpha(coef of 'temperature in April')

Y0=Y0.to_numpy()
a=np.dot(np.transpose(Y0-np.dot(Z,gamahat).reshape(n,1)),(D[:,0]-np.dot(Z,phihat)).reshape(n,1))
b=np.dot(D[:,0],(D[:,0]-np.dot(Z,phihat)).reshape(n,1))
alphahat0=a/b
print(alphahat0)
#which is negative, make sense

#CI of alphahat
alphahat=[alphahat0.item(),0,0]
sigma_sqaure=np.array((Y0-np.dot(Z,gamahat).reshape(n,1)-np.dot(D,alphahat).reshape(n,1))**2)
v_sqaure=(D[:,0]-np.dot(Z,phihat)).reshape(n,1)**2
Vhat=np.mean(sigma_sqaure*v_sqaure)/(np.mean(v_sqaure)**2)

#CI:
CI_L=alphahat[0]-np.sqrt(Vhat)*1.96/np.sqrt(n).astype("float64")
CI_U=alphahat[0]+np.sqrt(Vhat)*1.96/np.sqrt(n).astype("float64")
print("confidence interval for alpha is (",CI_L,",",CI_U,")")
#zero is in the interval, the effect is not clear