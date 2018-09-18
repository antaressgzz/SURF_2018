from oandapyV20 import API
import sqlite3
import v20
from datetime import datetime
import time

class test():
    
    def __init__(self,s, e, i):
        global hostname, port, ssl, ID, token
        
        hostname = "api-fxpractice.oanda.com"
        port = 443
        ssl = True
        ID = "101-011-8770085-001" #your ID
        token ="ee13bb5f4f3c6e7ae4062583ca668dc5-d378ce0d4ecd1876b902c786e154fa97" # your token
        
        global start, end, instrument, granularity
        start = s
        end = e
        instrument = i
        granularity = "H8"

        path = str('year/'+instrument+'_h8.db')
        # print(path)
        self.init_db(path)
        #insert_test(path)
        self.select(path)
        self.get_data_test(path) #passed
    
    def init_db(self,path):
        #with sqlite3.connect(DATABASE_DIR) as connection:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS MARKET \
                    (time TEXT, instrument VARCHAR(10), type CHAR(4), high FLOAT, \
                    close FLOAT, low FLOAT, open FLOAT, volume INTEGER, \
                    complete BOOLEAN)''')
        # print ("Table created successfully!")
        conn.close()
    
    def insert_test(self,path):
        conn = sqlite3.connect(path)
        conn.execute("INSERT INTO MARKET VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 ("2017-07-17T06:50:00.000000000Z", instrument, "ask", 1.14505, 1.14477, 1.14468, 1.14502, 91, True))
        conn.commit()
        print ("records created successfully!")
        conn.close()
    
    def insert(self,database_path, time, h, c, l, o, vol, complete, instrument="EUR_USD", price="mid"):
        conn = sqlite3.connect(database_path)
        conn.execute("INSERT INTO MARKET VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (time, instrument, price, h, c, l, o, vol, complete))
        conn.commit()
        # print ("success")
        conn.close()
    
    def select(self,path):
        conn = sqlite3.connect(path)
        cursor = conn.execute("SELECT time, instrument, high, volume, complete FROM MARKET")
        # for row in cursor:
            # print ("time = ", row[0])
            # print ("instrument = ", row[1])
            # print ("high = ", row[2])
            # print ("volume = ", row[3])
            # print ("complete = ", row[4], "\n")
        conn.close()
    
    def get_data_test(self,database_path):
        api = v20.Context(
                hostname,
                port,
                ssl,
                application="test",
                token=token)
        #kwargs = {}
        #test = None
        #if test is None:
        #    kwargs['price'] = "M"
        #    kwargs['granularity']= "M5"
        #    kwargs['smooth'] = True
        #    kwargs['fromTime'] = start
        #    kwargs['toTime'] = end
        response = api.instrument.candles(instrument, price="M", granularity="H8", smooth=True, fromTime=start, toTime=end)
    
        if response.status != 200:
            # print(response)
            # print(response.body)
            return
    
        candles = response.get("candles", 200)
        # print(candles)
    
        for candle in response.get("candles", 200):
            #print(candle.time) #unicode
            candletime = datetime.strptime(candle.time, "%Y-%m-%dT%H:%M:%S.000000000Z")
            unixtime = int(time.mktime(candletime.timetuple()))
            # print(unixtime)
            # print((candle.mid)) #candlestickdata
            # print(type(candle.mid.l)) #float
            #print(candle.price) #unicode
            # print(type(candle.volume)) #int
            # print(type(candle.complete)) #bool
            # print(type(candle.ask)) #None
            self.insert(database_path, candle.time, candle.mid.h, candle.mid.c, candle.mid.l, candle.mid.o, candle.volume, candle.complete, )
    
    
    


