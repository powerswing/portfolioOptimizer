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
    Parces stocks over given period from Yahoo. If end date (endDate) is not passed, then it is taken to be today

    Parameters
        stocks : list
            list of stocks to be parced
        startDate : datetime.datetime
            start day of desired period
        endDate : datetime.datetime
            end day of desired period

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

    def __init__(self, data, capital, assertNormality=True):
        self.data = data
        self.capital = capital
        self.alpha = 0.05
        self.simulationNumber = 10
        self.timeframe = 1
        self.percentile = 5
        self.riskfreeRate = 0
        self.assertNormality = assertNormality
        self.stocks = None
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
        self.weights = np.zeros(self.simulationNumber)
        self.weightedReturns = np.zeros(self.simulationNumber)
        self.weightedVariances = np.zeros(self.simulationNumber)
        self.weightedRisks = np.zeros(self.simulationNumber)
        self.sharpeRatios = np.zeros(self.simulationNumber)
        self.meanReturnsRealizations = None
        self.L = None
        self.Z = None
        self.returnsRealizations = None
        self.valueAtRisk = None
        self.conditionalValueAtRisk = None
        self.returnsRealizationsLast = None

    def getMoments(self):
        """
        Calculates empirical mean and covariance matrix of closing prices for given data. If assertNormality is True, asserts normality of returns for given stocks and calculates kurtosis. Those stocks that fail null hypothesis of returns normality are removed. Normality test is based D’Agostino and Pearson’s (1973)

        Parameters
            data : pd.DataFrame
                raw parsed data from yahoo with "Close" column to extract closing prices. To parse raw data from Yahoo, get_data_yahoo function of pandas_datareader.data must be used
            assertNormality : bool
                whether to trim data of stocks which returns fail normality test

        Returns
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
        # self.closingPricesM = self.closingPricesD.sort_index().resample(
        #     "M").apply(lambda x: x.iloc[-1, ])
        # self.closingPricesM = self.closingPricesD.resample("M").mean()

        self.pctReturns = self.closingPrices.pct_change().dropna()

        if self.assertNormality:
            self.statistic, self.pvalue = normaltest(self.pctReturns)
            self.kurtosis = kurtosis(self.pctReturns)
            self.normalityMask = self.pvalue < self.alpha
            self.stocks = [stock for (stock, mask) in zip(
                self.stocks, self.normalityMask) if mask]

            for i in [self.closingPrices, self.pctReturns]:

                i = i[
                    i.columns[self.normalityMask]]

        self.meanReturns = self.pctReturns.mean()
        self.covarianceMatrix = self.pctReturns.cov()

        return self.closingPrices, self.pctReturns, self.meanReturns, self.covarianceMatrix

    def initiateWeights(self):
        """
        Optimizes weights for given stocks using Sharpe ratio (Sharpe 1966) under normality assumption

        Parameters

        Returns


        """
        for simulation in range(self.simulationNumber):

            weight = np.random.random(len(self.stocks))
            weight /= np.sum(self.weights)
            self.weights[simulation] = weight

            weightedReturn = np.sum(self.meanReturns * weight)
            weightedVariance = np.dot(
                weight.T, np.dot(self.covarianceMatrix, weight))
            weightedStandardDeviation = np.sqrt(weightedVariance)

            self.weightedReturns[simulation] = weightedReturn
            self.weightedVariances[simulation] = weightedVariance
            self.weightedRisks[simulation] = weightedStandardDeviation

            sharpeRatio = (weightedReturn - self.riskfreeRate) / \
                weightedStandardDeviation
            self.sharpeRatios[simulation] = sharpeRatio

    def simulateReturns(self):
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
            timeframe : int, default is 9
                length of time path of each realization from today

        Returns
            returnsRealizations : np.ndarray
                matrix of simulations of returns

        """
        # matrix of means
        self.meanReturnsRealizations = np.full(
            shape=(self.timeframe, len(self.stocks)), fill_value=self.meanReturns).T

        self.returnsRealizations = np.full(
            shape=(self.timeframe, self.simulationNumber), fill_value=0.0)
        # Cholesky decomposition of covariance matrix
        self.L = np.linalg.cholesky(self.covarianceMatrix)

        for simulation in range(self.simulationNumber):
            self.Z = np.random.normal(
                size=(self.timeframe, len(self.stocks)))
            self.simulationReturns = self.meanReturnsRealizations + \
                np.inner(self.L, self.Z)
            self.returnsRealizations[:, simulation] = np.cumprod(
                np.inner(self.weights, self.simulationReturns.T) + 1) * self.capital

        return self.returnsRealizations

    def riskMetrics(self):
        """
        Returns the inverse of value-at-risk and conditional value-at-risk for given percentile level at the tail of simulation paths. To obtain the risk metrics, the return values are to be subtracted from capital

        Parameters
            capital : int
                value of investment
            returnsRealizations : np.ndarray
                matrix of simulations of returns
            percentile : float, default : 5 (translates to 95 percentile)
                percentile (aka confidence or percentile) level for loss cutoff

        Returns
            valueAtRisk : float
                value-at-risk for given confidence level
            conditionalValueAtRisk : float
                conditional value-at-risk for given confidence level
        """
        self.returnsRealizationsLast = pd.Series(
            self.returnsRealizations[-1, :])
        self.valueAtRisk = np.percentile(
            self.returnsRealizationsLast, self.percentile)
        self.conditionalValueAtRisk = self.returnsRealizationsLast[self.returnsRealizationsLast <= self.valueAtRisk].mean(
        )

        return self.capital - self.valueAtRisk, self.capital - self.conditionalValueAtRisk


# %%
data = getData(stocks, startDate, endDate)
po = portfolioOptimizer(data, capital)
closingPrices, pctReturns, meanReturns, covarianceMatrix = po.getMoments()
returnsRealizations = po.simulateReturns()
var, cvar = po.riskMetrics()
