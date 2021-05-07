# %%
# TODO: IMPLEMENT SHARPE RATIO
# https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
# https://codingandfun.com/portfolio-optimization-with-python/
# %%
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import normaltest

from pandas_datareader import data as pdr

plt.style.use(["seaborn", "fivethirtyeight"])

# %%

capital = 100
simulationNumber = 10
history = 300
timeframe = 30
stocks = ["DBAN.DE", "ADDYY"]
date1 = dt.datetime.now()
date0 = date1 - dt.timedelta(days=history)

# %%


class portfolioOptimizer(object):
    """
    #TODO: Add description
    """

    def __init__(self, stocks, capital, date0, date1, assertNormality=True):
        self.stocks = stocks
        self.capital = capital
        self.date0 = date0
        self.date1 = date1
        self.assertNormality = assertNormality
        self.alpha = 1e-3
        self.simulationNumber = 1000
        self.timeframe = 9
        self.tolerance = 5
        self.returnsRealizations = np.full(
            shape=(self.timeframe, self.simulationNumber), fill_value=0.0)
        self.data = None
        self.closingPrices = None
        self.pctReturns = None
        self.meanReturns = None
        self.covarianceMatrix = None
        self.normalityMask = None
        self.statistic = None
        self.pvalue = None
        self.weights = None
        self.meanReturnsRealizations = None
        self.L = None
        self.Z = None
        self.valueAtRisk = None
        self.conditionalValueAtRisk = None
        self.returnsRealizationsLast = None

    def getData(self):
        """
        Parces stocks over given period from Yahoo. If assertNormality is True, those stocks that fail null hypothesis of returns normality are removed. Normality test is based D’Agostino and Pearson’s (1973)

        Parameters
            stocks : list
                list of stocks to be parced
            date0 : int
                start day of desired period
            date1 : int
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
            self.stocks, self.date0, self.date1)
        self.closingPrices = self.data["Close"]

        self.pctReturns = self.closingPrices.pct_change().dropna()

        if self.assertNormality:
            self.statistic, self.pvalue = normaltest(self.pctReturns)
            self.normalityMask = self.pvalue < self.alpha
            self.stocks = [stock for (stock, mask) in zip(
                self.stocks, self.normalityMask) if not mask]
            self.closingPrices = self.closingPrices[self.closingPrices.columns[self.normalityMask]]
            self.pctReturns = self.pctReturns[self.pctReturns.columns[self.normalityMask]]

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
        # a matirx of returns averages
        self.meanReturnsRealizations = np.full(
            shape=(self.timeframe, len(self.stocks)), fill_value=self.meanReturns).T

        # Cholesky decomposition of covariance matrix
        self.L = np.linalg.cholesky(self.covarianceMatrix)

        # random weights
        self.weights = np.random.random(len(self.stocks))
        self.weights /= np.sum(self.weights)

        for simulation in range(self.simulationNumber):
            self.Z = np.random.normal(size=(self.timeframe, len(self.stocks)))
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
pO = portfolioOptimizer(stocks, capital, date0, date1)
closingPrices, pctReturns, meanReturns, covarianceMatrix = pO.getData()
returnsRealizations = pO.monteCarlo()
var, cvar = pO.riskMetrics()
