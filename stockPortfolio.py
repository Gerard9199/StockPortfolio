import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from sklearn.cluster import KMeans
from financeLib import monte_carlo, VaR, forecasting
#input
years = 12
today = date.today()
start = today.replace(year=today.year - years)

composite = '^IXIC' #Nasdaq composite
csv = pd.read_csv('stock.csv') #Nasdaq stocks
stockList = csv['Stock'].tolist()

stockPrice = yf.download(stockList, start=start, end=today, interval='1d')['Adj Close']
stockPrice['Composite'] = yf.download(composite, start=start, end=today, interval='1d')['Adj Close']
stockPrice = stockPrice.fillna(method='ffill')
stockPrice = stockPrice.fillna(0)

returns = stockPrice.pct_change()
reu = returns.fillna(0)
reu = reu.replace(np.inf, 0)
reu = reu.mean()
beta = (returns.cov()['Composite'])/(returns['Composite'].var())
stockSummary = pd.concat([beta, reu], axis=1)
stockSummary = stockSummary.rename(columns={'Composite': 'Beta',0: 'Returns'})
composite_summary = stockSummary.loc['Composite']
stockSummary = stockSummary.drop(['Composite'], axis=0)
stockSummary = stockSummary.dropna(axis=0)

z = 8
wcss = []
for i in range(1, z):
    kmeans = KMeans(n_clusters=i, max_iter=300)
    kmeans.fit(stockSummary)
    wcss.append(kmeans.inertia_)
clustering = KMeans(n_clusters=z, max_iter=300)
clustering.fit(stockSummary)
stockSummary['Clusters'] = clustering.labels_

print(stockSummary['Clusters'].value_counts())

print(stockSummary.groupby('Clusters').mean())


higherReturn = ((stockSummary.groupby('Clusters').mean())['Returns'].sort_values()[1:]).index[0]
selectedCluster = stockSummary[stockSummary['Clusters'] == higherReturn]
confidenceInterval = 0.3
portfolio = selectedCluster[selectedCluster['Returns'] > (selectedCluster.mean()['Returns'] - (selectedCluster.std()['Returns']))]
portfolioTicker = (portfolio.index).tolist()
portfolioTicker = np.random.choice(portfolioTicker, 5, replace=False)
print(portfolioTicker)

portfolioReturn = (stockPrice[portfolioTicker].pct_change()).dropna()
portfolioReturn = portfolioReturn.replace(np.inf, 0)
lastPrice = stockPrice[portfolioTicker].iloc[-1]

monteCarlo = monte_carlo(portfolioReturn, portfolioTicker)
VaRReturns = monteCarlo[-1,:]


forec = []
for i in range(0, len(portfolioTicker)):
    print(portfolioTicker[i:i+1][0])
    vars()[portfolioTicker[i:i+1][0]] = forecasting(stockPrice[portfolioTicker[i:i+1]], portfolioTicker[i:i+1][0])
    forec.append(vars()[portfolioTicker[i:i+1][0]])
forecast = pd.concat(forec, axis=1)

forecastedReturn = (forecast.loc[:, ::2]).pct_change().dropna()
forecastedMonteC = monte_carlo(forecastedReturn, portfolioTicker)
forecastedVaRRet = forecastedMonteC[-1,:]

print('With your portfolio')
portfolioVaR = VaR(VaRReturns)

print('With your forecasted portfolio')
forecastedVaR = VaR(forecastedVaRRet)
