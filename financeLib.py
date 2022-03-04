# ==============================================================
# Author: Gerardo Álvarez
# Instagram: @QuantJerry
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script has been originally created by Gerardo Álvarez.
# Any explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ==============================================================

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def monte_carlo(returns, tickers, simulations=10000, capital=10000, t=100):
    weights = np.random.random(len(tickers)) #the weigth of each stock in the portfolio, the sum will be 1
    weights /= np.sum(weights)
    meanMatrix = (np.full(shape=(t, len(tickers)), fill_value=returns.mean())).T
    portfolioSimulations = np.full(shape=(t, simulations), fill_value=0)
    for i in range(0, simulations):
        zFactor = np.random.normal(size=(t, len(tickers)))
        Cholesky = np.linalg.cholesky(returns.cov())
        dailyReturns = meanMatrix + np.inner(Cholesky, zFactor)
        portfolioSimulations[:, i] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*capital
    return portfolioSimulations

def VaR(returns, capital=10000, confidence_interval=95):
    VaR = capital - np.percentile(returns, 100-confidence_interval)
    CVaR = capital - returns[returns <= np.percentile(returns, 100-confidence_interval)].mean()
    print('VaR ${}'.format(round(VaR, 2)))
    print('CVaR ${}'.format(round(CVaR, 2)))

def forecasting(stockValues, stockTicker):
    data = stockValues.values
    dataLenght = round(len(data)*.8)#80% of the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(data)
    trainData = scaledData[0:dataLenght, :]
    xTrain, yTrain = [], []
    for i in range(60, len(trainData)):
        xTrain.append(trainData[i-60:i, 0])
        yTrain.append(trainData[i, 0])
    xTrain, yTrain = np.array(xTrain), np.array(yTrain)
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xTrain, yTrain, batch_size=1, epochs=1)
    testData = scaledData[dataLenght - 60:,:]
    xTest, yTest = [], data[dataLenght:,:]
    for i in range(60, len(testData)):
        xTest.append(testData[i-60:i, 0])
    xTest = np.array(xTest)
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
    forecasting = model.predict(xTest)
    forecasting = scaler.inverse_transform(forecasting)
    rmse = np.sqrt(np.mean(forecasting-yTest)**2)
    train = data[:dataLenght]
    valid= pd.DataFrame(data[dataLenght:], columns=['Original {}'.format(stockTicker)])
    valid['Forecast {}'.format(stockTicker)] = forecasting
    return valid
