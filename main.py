# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:08:37 2018

@author: Andrew
"""
import seaborn as sns
import pandas as pd
import numpy as np
#from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm, datasets
from sklearn.kernel_ridge import KernelRidge

houses = fetch_california_housing()
digits = datasets.load_iris()

data = houses.data
names = houses.feature_names
target = houses.target
#Q1
#DistPlots for all 8 features, individually
#sns.distplot(data[:,0], axlabel=names[0])
#sns.distplot(data[:,1], axlabel=names[1])
#sns.distplot(data[:,2], axlabel=names[2])
#sns.distplot(data[:,3], axlabel=names[3])
#sns.distplot(data[:,4], axlabel=names[4])
#sns.distplot(data[:,5], axlabel=names[5])
#sns.distplot(data[:,6], axlabel=names[6])
#sns.distplot(data[:,7], axlabel=names[7])

#Target DistPlot
#sns.distplot(houses.target, axlabel='Target')

test = max(data[:,2])
test2 = max(data[:,5])

housingDF = pd.DataFrame(data=data, columns=names)
#All 8 DistPlots together
#fig1 = housingDF.hist(bins=40, figsize=(9, 6))

#fig, axs = plot_partial_dependence()

#Q3
X_train, X_test, y_train, y_test = train_test_split(data, target)

#linear regression
lin = LinearRegression().fit(X_train, y_train)
print("Linear Score: ", lin.score(X_test, y_test))

#Ridge regression w/ CV
rid = RidgeCV().fit(X_train, y_train)
print("Ridge Score: ", rid.score(X_test, y_test))

#Lasso regression w/ CV
lasso = LassoCV().fit(X_train, y_train)
print("Lasso Score: ", lasso.score(X_test, y_test))

#Elastic Net regression w/ CV
ela = ElasticNetCV().fit(X_train, y_train)
print("ElasticNet Score: ", ela.score(X_test, y_test))

#Using StandardScaler
scaler = StandardScaler()
dataSTD = scaler.fit_transform(data, target)
X_train2, X_test2, y_train2, y_test2 = train_test_split(dataSTD, target)
print("")
print("With Standardization:")

#linear regression STD
lin = LinearRegression().fit(X_train2, y_train2)
print("Linear Score: ", lin.score(X_test2, y_test2))

#Ridge regression w/ CV STD 
rid = RidgeCV().fit(X_train2, y_train2)
print("Ridge Score: ", rid.score(X_test2, y_test2))

#Lasso regression w/ CV STD
lasso = LassoCV().fit(X_train2, y_train2)
print("Lasso Score: ", lasso.score(X_test2, y_test2))

#Elastic Net regression w/ CV STD
ela = ElasticNetCV().fit(X_train2, y_train2)
print("ElasticNet Score: ", ela.score(X_test2, y_test2))

#Q4
print("")
estimator = Ridge()
paramsR = {'alpha': [25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01],
          'fit_intercept': [True, False],
          }
gsCVR =  GridSearchCV(estimator, paramsR)
gsCVR.fit(X_train, y_train)
#print(gsCVR.best_params_)
rid = Ridge(alpha=25, fit_intercept=True).fit(X_train, y_train)
print("Ridge Score(w/ best parameters): ", rid.score(X_test, y_test))
estimator = LassoCV()
paramsL = {'cv': [3,4,5,6],
          'fit_intercept': [True, False],
          'normalize': [True, False],
          'precompute': [True, False]
          }
gsCVL =  GridSearchCV(estimator, paramsL)
gsCVL.fit(X_train, y_train)
#print(gsCVL.best_params_)
las = LassoCV(cv=3, fit_intercept=True, normalize=True, precompute=True).fit(X_train, y_train)
print("Lasso Score(w/ best parameters): ", las.score(X_test, y_test))
estimator = ElasticNetCV()
paramsL = {'cv': [3,4,5,6],
          'normalize': [True, False],
          'precompute': [True, False]
          }
gsCVE =  GridSearchCV(estimator, paramsL)
gsCVE.fit(X_train, y_train)
#print(gsCVE.best_params_)
en = ElasticNetCV(cv=3, normalize=False, precompute=True).fit(X_train, y_train)
print("ElasticNet Score(w/ best parameters): ", en.score(X_test, y_test))

