import test
import datetime

# 375天的数据， 15天／ep，25个ep, 2017-07-01 to 2018-07-11
startdate="2015-04-01"
start = datetime.datetime.strptime(startdate, '%Y-%m-%d')
episodes=97
days_per_ep=15
for ep in range(episodes):
    delta=datetime.timedelta(days=ep*15)
    newdate=start+delta
    new=newdate.strftime('%Y-%m-%d')

    deltatwo=datetime.timedelta(days=(ep+1)*15)
    enddate=start+deltatwo
    end=enddate.strftime('%Y-%m-%d')
    instrument="USD_CAD"
    #USD_JPY EUR_USD GBP_USD USD_CAD

    DataColr = test.test(new, end,instrument ,"M30",str('csvDocu/30M/'+instrument+'_M30.db'))



print('saved!')


#
