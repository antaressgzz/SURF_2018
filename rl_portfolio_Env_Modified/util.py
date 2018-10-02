import numpy as np
from .config import eps
# import talib

def sharpe(pc_array):
    """calculate sharpe ratio with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: sharpe ratio
    """
    pc_array = pc_array-1.0
    return np.mean(pc_array)/np.std(pc_array)
def MDD(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)


def softmax(w, t=1.0):
    """softmax implemented in numpy."""
    log_eps = np.log(eps)
    w = np.clip(w, log_eps, -log_eps)  # avoid inf/nan
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def sigmoid(x):
    return 1 / (1+np.exp(-0.25*x))

def talib_features(close_array):  # input : [asset,length]
    close_array = np.transpose(close_array)  # [length, asset]
    newF = np.ones([np.shape(close_array)[1], np.shape(close_array)[0] - 50, 5])
    for i in range(np.shape(close_array)[1]):
        close = np.transpose(close_array)[:][i]
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # print(macd, macdsignal)
        upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        macd = macd.tolist()
        macdsignal = macdsignal.tolist()
        upperband = upperband.tolist()
        middleband = middleband.tolist()
        lowerband = lowerband.tolist()
        newfeactures = [macd, macdsignal, upperband, middleband, lowerband]
        newfeactures = list(map(list, zip(*newfeactures)))
        newfeactures = newfeactures[50:][:]
        newF[i][:] = newfeactures
    return newF  # output : [asset, length-50, talib]
