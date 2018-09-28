

def NewFeactures(close_array):  # input : [asset,length]

    close_array=np.transpose(close_array)
    newF=[]
    for i in range(np.shape(close_array)[1]):
        close=np.transpose(close_array[:][i])
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # print(macd, macdsignal)
        upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        macd = macd.tolist();
        macdsignal = macdsignal.tolist();
        upperband = upperband.tolist();
        middleband = middleband.tolist();
        lowerband = lowerband.tolist()
        newfeactures=[]
        newfeactures = [macd, macdsignal, upperband, middleband, lowerband]
        newfeactures = map(list, zip(*newfeactures))
        newfeactures = newfeactures[50:][:]
        newF[i][:]=newfeactures
    return newF # output : [asset, length, talib]

if __name__=='__main__':
    import talib
    import numpy as np

    print(dir(talib))
#    df=np.random.rand(3,100)
#    print(talib.get_functions())
    #NewFeactures(df)