# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:03:02 2019

@author: tejas
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:,1: ]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediction on Test Set
y_pred = regressor.predict(X_test)

#Backward Elimination
import statsmodels.api as sm
X = np.append( arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, (0,1,2,3,4,5)]
regressor_ols = sm.OLS(y, X_opt).fit()
regressor_ols.summary()

X_opt = X[:, (0,1,3,4,5)]
regressor_ols = sm.OLS(y, X_opt).fit()
regressor_ols.summary()

X_opt = X[:, (0,3,4,5)]
regressor_ols = sm.OLS(y, X_opt).fit()
regressor_ols.summary()

X_opt = X[:, (0,3,5)]
regressor_ols = sm.OLS(y, X_opt).fit()
regressor_ols.summary()

X_opt = X[:, (0,3)]
regressor_ols = sm.OLS(y, X_opt).fit()
regressor_ols.summary()
#R&D funds are the deciding factor for the Profit for the company