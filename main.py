# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:08:37 2018

@author: Andrew
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm, datasets
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence


def main(): 
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
    
    print("")
    print("Dependency on Targets: ")
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, loss='huber',random_state=1)
    clf.fit(data,target)
    feat = [0,1,2,3,4,5,6,7]
    '''fig, axs = plot_partial_dependence(clf, data, feat, feature_names=names,n_jobs=3, grid_resolution=50)
    fig.suptitle('Dependence of the target on each feature: ')
    plt.subplots_adjust(top=0.9, wspace=0.6, hspace=0.6)
    plt.show()'''
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
    param_range = np.logspace(-3,7,200)
    train_scores, test_scores = validation_curve(Ridge(), data, target, "alpha", param_range=param_range, cv=5)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.title("Validation Curve with Ridge")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
    alphas = np.logspace(-3, 7, 200)
    
    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(data, target)
        coefs.append(ridge.coef_)
    
    ax = plt.gca()
    
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients of each feature')
    plt.axis('tight')
    plt.legend()
    plt.show()
    
    
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
    param_range = np.logspace(-7,3,200) 
    train_scores, test_scores = validation_curve(Lasso(), data, target, "alpha", param_range=param_range, cv=5)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.title("Validation Curve with Lasso")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
    '''alphas = np.logspace(-7, 3, 200)
    
    coefs = []
    for a in alphas:
        lasso1 = Lasso(alpha=a, fit_intercept=False)
        lasso1.fit(data, target)
        coefs.append(lasso1.coef_)
    
    ax = plt.gca()
    
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients of each feature')
    plt.axis('tight')
    plt.legend()
    plt.show()'''
    
    las = LassoCV(cv=3, fit_intercept=True, normalize=True, precompute=True).fit(X_train, y_train)
    print("Lasso Score(w/ best parameters): ", las.score(X_test, y_test))
    estimator = ElasticNetCV()
    paramsL = {'cv': [3,4,5,6],
              'normalize': [True, False],
              'precompute': [True, False]
              }
    gsCVE =  GridSearchCV(estimator, paramsL)
    gsCVE.fit(X_train, y_train)
    
    train_scores, test_scores = validation_curve(ElasticNet(), data, target, "alpha", param_range=param_range, cv=3)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.title("Validation Curve with ElasticNet")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
    '''alphas = np.logspace(-7, 3, 200)
    
    coefs = []
    for a in alphas:
        eN1 = ElasticNet(alpha=a, fit_intercept=False)
        eN1.fit(data, target)
        coefs.append(eN1.coef_)
    
    ax = plt.gca()
    
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('ElasticNet coefficients of each feature')
    plt.axis('tight')
    plt.legend()
    plt.show()'''
    
    #print(gsCVE.best_params_)
    en = ElasticNetCV(cv=3, normalize=False, precompute=True).fit(X_train, y_train)
    print("ElasticNet Score(w/ best parameters): ", en.score(X_test, y_test))
    
if __name__ == '__main__':
    main()