# %%
# TODO: IMPLEMENT SHARPE RATIO
# TODO: TEST NORMALITY OF STOCK RETURNS
# https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
# https://codingandfun.com/portfolio-optimization-with-python/
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from pandas_datareader import data as pdr

plt.style.use(["seaborn", "fivethirtyeight"])

# %%

face = 100
simulationNumber = 10
history = 300
timeframe = 30
stocks = ["DBAN.DE", "ADDYY"]
date1 = dt.datetime.now()
date0 = date1 - dt.timedelta(days=history)

weights = np.random.random(len(stocks))
weights /= np.sum(weights)

# %%


def plotSimulation(returnsRealization):
    plt.plot(returnsRealization, )
    plt.ylabel("Cumulative Returns")
    plt.xlabel("Days")
    plt.title("Monte Carlo Simulation for Stock Returns")
    plt.show()


def monteCarlo(stocks, face, meanReturns, covarianceMatrix, simulationNumber, timeframe):
    """
    Implements Normal Carlo Simulation using Cholesky Decomposition for covariance matrix. Function returns matrix of simulated returns given by S = M + Z*L where M is matrix of empirical average relative returns, Z is multivariate normal matrix and L is lower triangular matrix of Cholesky decomposition of empirical covariance matrix, and * is the inner product operator

    Parameters
        stocks : list
            list of stocks. Its length is used to define the shape of multivariate normal distribution
        face : int
            value of investment
        meanReturns : pd.Series
            average of relative returns per given stock
        covarianceMatrix : pd.DataFrame
            covariance matrix of relative returns for given stocks
        simulationNumber : int
            number of simulations to run
        timeframe : int
            length of time path of each realization from today

    Returns
        returnsRealization : np.ndarray
            matrix of simulations of returns

    """
    # a matirx of returns averages
    meanReturnsRealization = np.full(
        shape=(timeframe, len(stocks)), fill_value=meanReturns).T

    # zero-matrix updated with random realization after each simulation
    returnsRealization = np.full(
        shape=(timeframe, simulationNumber), fill_value=0.0)

    # Cholesky decomposition of covariance matrix
    L = np.linalg.cholesky(covarianceMatrix)

    for simulation in range(simulationNumber):
        Z = np.random.normal(size=(timeframe, len(stocks)))
        simulationReturns = meanReturnsRealization + np.inner(L, Z)
        returnsRealization[:, simulation] = np.cumprod(
            np.inner(weights, simulationReturns.T) + 1) * face

    return returnsRealization


def stockMoments(stocks, date0, date1):
    """
    A Yahoo parcer of closing prices of given stocks for given period

    Parameters
        stocks : list
            list of stocks to be parced
        date0 : int
            start day of desired period
        date1 : int
            end day of desired period

    Returns
        data : pd.DataFrame
            closing prices of given stocks for given period
        meanReturns : pd.Series
            average of relative returns per given stock
        covarianceMatrix : pd.DataFrame
            covariance matrix of relative returns for given stocks

    """
    data = pdr.get_data_yahoo(stocks, date0, date1)
    data = data["Close"]

    # we apply log to allow additivity between percent changes
    logpctReturns = data.pct_change().apply(lambda x: np.log(1+x))
    meanReturns = logpctReturns.mean()
    covarianceMatrix = logpctReturns.cov()

    return data, meanReturns, covarianceMatrix


def riskMetrics(face, returnsRealization, tolerance=5):
    """
    Returns value-at-risk and conditional value-at-risk for given tolerance level at the tail of simulation path

    Parameters
        face : int
            value of investment
        returnsRealization : np.ndarray
            matrix of simulations of returns
        tolerance : float, default : 1 (translates to 99 percentile)
            tolerance (aka confidence or percentile) level for loss cutoff

    Returns
        valueAtRisk : float
            value-at-risk for given confidence level
        conditionalValueAtRisk : float
            conditional value-at-risk for given confidence level
    """
    returnsRealizationLast = pd.Series(returnsRealization[-1, :])
    valueAtRisk = np.percentile(returnsRealizationLast, tolerance)
    conditionalValueAtRisk = returnsRealizationLast[returnsRealizationLast <= valueAtRisk].mean(
    )

    return face - valueAtRisk, face - conditionalValueAtRisk


# %%
_, meanReturns, covarianceMatrix = stockMoments(stocks, date0, date1)
returnsRealization = monteCarlo(
    stocks, face, meanReturns, covarianceMatrix, simulationNumber, timeframe)
plotSimulation(returnsRealization)

# %%
VaR, cVaR = riskMetrics(face, returnsRealization, 1)
print(VaR, cVaR)
# %%
