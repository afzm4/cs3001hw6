# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:08:37 2018

@author: Andrew
"""
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

houses = fetch_california_housing()

data = houses.data
names = houses.feature_names
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
fig1 = housingDF.hist(bins=40, figsize=(9, 6))