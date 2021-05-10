# %%
# TODO: IMPLEMENT SHARPE RATIO
# https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
# https://codingandfun.com/portfolio-optimization-with-python/
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import normaltest
from scipy.stats import kurtosis

from pandas_datareader import data as pdr

plt.style.use(["seaborn", "fivethirtyeight"])

# %%
capital = 100
history = 365
stocks = ["DBAN.DE", "ADDYY"]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=history)
# %%


def getData(stocks, startDate, endDate=dt.datetime.now()):
    """
    Parces stocks over given period from Yahoo

    Parameters
        stocks : list
            list of stocks to be parced
        startDate : datetime.datetime
            start date of desired period
        endDate : datetime.datetime, default is today
            end date of desired period

    Returns
        data : pd.DataFrame
            parsed data
    """
    data = pdr.get_data_yahoo(
        stocks, startDate, endDate)

    return data


class portfolioOptimizer(object):
    """
    # TODO: Add description
    """

    def __init__(self, data, capital, simulationNumber=10):
        self.data = data
        self.capital = capital
        self.simulationNumber = simulationNumber
        self.stocks = None
        self.stocksNumber = None
        self.returnsRealizations = None
        self.closingPrices = None
        self.pctReturns = None
        self.meanReturns = None
        self.covarianceMatrix = None
        self.statistic = None
        self.pvalue = None
        self.kurtosis = None
        self.normalityMask = None
        self.optimalWeights = None
        self.weights = None
        self.weightedReturns = None
        self.weightedVariances = None
        self.weightedRisks = None
        self.sharpeRatios = None
        self.bestWeight = None
        self.meanReturnsRealizations = None
        self.L = None
        self.Z = None
        self.returnsRealizations = None
        self.valueAtRisk = None
        self.conditionalValueAtRisk = None
        self.returnsRealizationsLast = None

    def getMoments(self, assertNormality=True, alpha=0.05):
        """
        Calculates empirical mean and covariance matrix of closing prices for given data. If assertNormality is True, asserts normality of returns for given stocks and calculates kurtosis. Those stocks that fail null hypothesis of returns normality are removed. Normality test is based D’Agostino and Pearson’s (1973)

        Parameters
            assertNormality : bool
                whether to trim data of stocks which returns fail normality test
            alpha : float
                critical value of chi-squared distribution

        Stores
            closingPrices : pd.DataFrame
                closing prices of parsed data (NaNs are dropped)
            pctReturns : pd.Series
                percent changes of closing prices
            meanReturns : pd.Series
                average of percentage returns per given stock
            covarianceMatrix : pd.DataFrame
                covariance matrix of percentage returns for given stocks
        """
        self.closingPrices = self.data["Close"]
        self.stocks = self.closingPrices.columns.values
        self.stocksNumber = len(self.stocks)
        # self.closingPricesM = self.closingPricesD.sort_index().resample(
        #     "M").apply(lambda x: x.iloc[-1, ])
        # self.closingPricesM = self.closingPricesD.resample("M").mean()

        self.pctReturns = self.closingPrices.pct_change().dropna()

        if assertNormality:
            self.statistic, self.pvalue = normaltest(self.pctReturns)
            self.kurtosis = kurtosis(self.pctReturns)
            self.normalityMask = self.pvalue < alpha
            self.stocks = [stock for (stock, mask) in zip(
                self.stocks, self.normalityMask) if mask]
            self.stocksNumber = len(self.stocks)

            for i in [self.closingPrices, self.pctReturns]:

                i = i[
                    i.columns[self.normalityMask]]

        self.meanReturns = self.pctReturns.mean()
        self.covarianceMatrix = self.pctReturns.cov()

    def simulateWeights(self, riskfreeRate=0):
        """
        Optimizes weights for given stocks using Sharpe ratio (Sharpe 1966) under assumption of returns normality

        Parameters
            riskfreeRate : float, default is 0
                Risk-free rate of return used to calculate Sharpe ratio
        Stores
            #TODO


        """
        self.weights = np.zeros(
            [self.simulationNumber, self.stocksNumber])
        self.weightedReturns = np.zeros(
            [self.simulationNumber, self.stocksNumber])
        self.weightedVariances = np.zeros(
            [self.simulationNumber, self.stocksNumber])
        self.weightedRisks = np.zeros(
            [self.simulationNumber, self.stocksNumber])
        self.sharpeRatios = np.zeros(
            [self.simulationNumber, self.stocksNumber])
        self.bestWeight = np.zeros(
            [1, self.stocksNumber])

        for simulation in range(self.simulationNumber):

            weight = np.random.random(self.stocksNumber)
            weight /= np.sum(weight)
            self.weights[simulation] = weight

            weightedReturn = np.sum(self.meanReturns * weight)
            weightedVariance = np.dot(
                weight.T, np.dot(self.covarianceMatrix, weight))
            weightedStandardDeviation = np.sqrt(weightedVariance)

            self.weightedReturns[simulation] = weightedReturn
            self.weightedVariances[simulation] = weightedVariance
            self.weightedRisks[simulation] = weightedStandardDeviation

            sharpeRatio = (weightedReturn - riskfreeRate) / \
                weightedStandardDeviation
            self.sharpeRatios[simulation] = sharpeRatio

        self.bestWeight = self.weights[np.argmax(self.sharpeRatios[:, 0])]

        #self.bestWeight = self.weights[np.argmax(po.sharpeRatios)]

    def simulateReturns(self, timeframe=1):
        """
        Implements Normal Carlo simulation (f.e. Raychaudhuri 2008) using Cholesky Decomposition (f.e. Highham 2009) for covariance matrix. Function returns matrix of simulated returns given by S = M + Z*L where M is matrix of empirical average relative returns, Z is multivariate normal matrix and L is lower triangular matrix of Cholesky decomposition of empirical covariance matrix, and * is the inner product operator

        Parameters
            stocks : list
                list of stocks. Its length is used to define the shape of multivariate normal distribution
            capital : int
                value of investment
            meanReturns : pd.Series
                average of relative returns per given stock
            covarianceMatrix : pd.DataFrame
                covariance matrix of relative returns for given stocks
            simulationNumber : int, default is 1000
                number of simulations to run
            timeframe : int, default is 1
                length of time path of each realization from today

        Stores
            returnsRealizations : np.ndarray
                matrix of simulations of returns

        """
        # matrix of means
        self.meanReturnsRealizations = np.full(
            shape=(timeframe, self.stocksNumber), fill_value=self.meanReturns).T

        self.returnsRealizations = np.full(
            shape=(timeframe, self.simulationNumber), fill_value=0.0)
        # Cholesky decomposition of covariance matrix
        self.L = np.linalg.cholesky(self.covarianceMatrix)

        for simulation in range(self.simulationNumber):
            self.Z = np.random.normal(
                size=(timeframe, self.stocksNumber))
            self.simulationReturns = self.meanReturnsRealizations + \
                np.inner(self.L, self.Z)
            self.returnsRealizations[:, simulation] = np.cumprod(
                np.inner(self.bestWeight, self.simulationReturns.T) + 1) * self.capital

    def riskMetrics(self, percentile=5):
        """
        Returns the inverse of value-at-risk and conditional value-at-risk for given percentile level at the tail of simulation paths. To obtain the risk metrics, the return values are to be subtracted from capital

        Parameters
            capital : int
                value of investment
            returnsRealizations : np.ndarray
                matrix of simulations of returns
            percentile : float, default is 5 (translates to 95 percentile)
                percentile (aka confidence or percentile) level for loss cutoff

        Stores
            valueAtRisk : float
                value-at-risk for given confidence level
            conditionalValueAtRisk : float
                conditional value-at-risk for given confidence level
        """
        self.returnsRealizationsLast = pd.Series(
            self.returnsRealizations[-1, :])
        self.valueAtRisk = np.percentile(
            self.returnsRealizationsLast, percentile)
        self.conditionalValueAtRisk = self.returnsRealizationsLast[self.returnsRealizationsLast <= self.valueAtRisk].mean(
        )

        self.valueAtRisk = self.capital - self.valueAtRisk
        self.conditionalValueAtRisk = self.capital - self.conditionalValueAtRisk


# %%
data = getData(stocks, startDate, endDate)
po = portfolioOptimizer(data, capital)
po.getMoments()
po.simulateWeights()
po.simulateReturns()
po.riskMetrics()
