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
stocks = ["DBAN.DE", "ADDYY"]
# %%


class portfolioOptimizer(object):
    """
    #TODO: Add description
    """

    def __init__(self, stocks, capital, assertNormality=True):
        self.stocks = stocks
        self.capital = capital
        self.history = 365
        self.alpha = 0.05
        self.simulationNumber = 10
        self.timeframe = 1
        self.tolerance = 5
        self.currentTimestamp = dt.datetime.now()
        self.historyTimestamp = self.currentTimestamp - \
            dt.timedelta(days=self.history)
        self.assertNormality = assertNormality
        self.returnsRealizations = None
        self.data = None
        self.closingPrices = None
        self.pctReturns = None
        self.meanReturns = None
        self.covarianceMatrix = None
        self.statistic = None
        self.pvalue = None
        self.kurtosis = None
        self.normalityMask = None
        self.weights = None
        self.meanReturnsRealizations = None
        self.L = None
        self.Z = None
        self.returnsRealizations = None
        self.valueAtRisk = None
        self.conditionalValueAtRisk = None
        self.returnsRealizationsLast = None

    def getData(self):
        """
        Parces stocks over given period from Yahoo. If assertNormality is True, those stocks that fail null hypothesis of returns normality are removed. Normality test is based D’Agostino and Pearson’s (1973)

        Parameters
            stocks : list
                list of stocks to be parced
            historyTimestamp : int
                start day of desired period
            currentTimestamp : int
                end day of desired period
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
        self.data = pdr.get_data_yahoo(
            self.stocks, self.historyTimestamp, self.currentTimestamp)
        self.closingPrices = self.data["Close"]
        # self.closingPricesM = self.closingPricesD.sort_index().resample(
        #     "M").apply(lambda x: x.iloc[-1, ])
        #self.closingPricesM = self.closingPricesD.resample("M").mean()

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

    def monteCarlo(self):
        """
        Implements Normal Carlo Simulation using Cholesky Decomposition for covariance matrix. Function returns matrix of simulated returns given by S = M + Z*L where M is matrix of empirical average relative returns, Z is multivariate normal matrix and L is lower triangular matrix of Cholesky decomposition of empirical covariance matrix, and * is the inner product operator

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

        # random weights
        self.weights = np.random.random(len(self.stocks))
        self.weights /= np.sum(self.weights)

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
        Returns the inverse of value-at-risk and conditional value-at-risk for given tolerance level at the tail of simulation paths. To obtain the risk metrics, the return values are to be subtracted from capital

        Parameters
            capital : int
                value of investment
            returnsRealizations : np.ndarray
                matrix of simulations of returns
            tolerance : float, default : 1 (translates to 99 percentile)
                tolerance (aka confidence or percentile) level for loss cutoff

        Returns
            valueAtRisk : float
                value-at-risk for given confidence level
            conditionalValueAtRisk : float
                conditional value-at-risk for given confidence level
        """
        self.returnsRealizationsLast = pd.Series(
            self.returnsRealizations[-1, :])
        self.valueAtRisk = np.percentile(
            self.returnsRealizationsLast, self.tolerance)
        self.conditionalValueAtRisk = self.returnsRealizationsLast[self.returnsRealizationsLast <= self.valueAtRisk].mean(
        )

        return self.capital - self.valueAtRisk, self.capital - self.conditionalValueAtRisk


# %%
po = portfolioOptimizer(stocks, capital)
closingPrices, pctReturns, meanReturns, covarianceMatrix = po.getData()
returnsRealizations = po.monteCarlo()
var, cvar = po.riskMetrics()

# %%
