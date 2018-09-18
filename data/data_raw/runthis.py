import test
import datetime

# 375天的数据， 15天／ep，25个ep, 2017-07-01 to 2018-07-11
startdate="2015-04-01"
start = datetime.datetime.strptime(startdate, '%Y-%m-%d')
episodes=150
days_per_ep=15
for ep in range(episodes):
    delta=datetime.timedelta(days=ep*15)
    newdate=start+delta
    new=newdate.strftime('%Y-%m-%d')

    deltatwo=datetime.timedelta(days=(ep+1)*15)
    enddate=start+deltatwo
    end=enddate.strftime('%Y-%m-%d')

    DataColr = test.test(new, end, "USD_JPY")



print('saved!')


#