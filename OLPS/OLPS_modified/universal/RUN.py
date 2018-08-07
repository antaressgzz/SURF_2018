from universal.algo import Algo
import numpy as np
from universal import tools
import pandas as pd
from universal import tools
from universal import algos
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging

class RUN():
    # data = tools.dataset('nyse_o')

    def __init__(self,df,algo):
        #self.data = tools.dataset(path,'train')
        self.data = df
        self.data=pd.DataFrame(self.data)
        #self.AlgosCalcu(algo)
        self.options = algo
        self.result=[]
        self.res=[]
        self.sharpCollect=[]
        self.MDDCollect=[]


    #
    # algo = algos.OLMAR(window=5, eps=10)
    # result = algo.run(data)
    # print(result.summary())
    # result.plot(weights=False, assets=True,portfolio_label='olmar', ucrp=True, logy=True) # ucrp--(uniform constant rebalan

    def AlgosCalcu(self):
        self.options=self.options.split(',')

        for i in self.options:
            self.result.append(self.AlgResult(i))

        #self.AlgoPlot(result,self.options)

    def AlgResult(self,ag):

        if ag=='OLMAR':
            algo = algos.OLMAR(window=5, eps=10)
        if ag=='CRP':
            algo = algos.CRP()
        if ag=='BAH':
            algo = algos.BAH()
        if ag=='Anticor':
            algo = algos.Anticor(window=5)
        if ag=='CORN':
            algo = algos.CORN(window=5)
        if ag=='BCRP':
            algo = algos.BCRP()
        if ag=='CWMR':
            algo = algos.CWMR(eps=10)
        if ag == 'PAMR':
            algo = algos.PAMR(eps=10)
        if ag == 'RMR':
            algo = algos.RMR(window=5,eps=10)
        if ag == 'UP':
            algo = algos.UP()
        if ag == 'WMAMR':
            algo = algos.WMAMR(window=5)
        if ag == 'ONS':
            algo = algos.ONS()
        if ag == 'Kelly':
            algo = algos.Kelly(window=5)
        if ag == 'EG':
            algo = algos.EG()
        if ag == 'BNN':
            algo = algos.BNN()
        if ag == 'BestSoFar':
            algo = algos.BestSoFar()
        if ag == 'BestMarkowitz':
            algo = algos.BestMarkowitz()

        result = algo.run(self.data)
        return result

    def return_df(self):
        axs=[]
        for i,result in enumerate(self.result):
            self.res.append(result.plot(weights=False, assets=False,portfolio_label=self.options[i], ucrp=True, logy=True)) # ucrp--(uniform constant rebalanced portfolio)
            # print(res)
            self.sharpCollect.append(result.sharpMDD()[0])
            self.MDDCollect.append(result.sharpMDD()[1])
            #plt.plot(self.x,res)
        return self.res, self.options, self.sharpCollect, self.MDDCollect

        # plt.legend(options)
        # plt.grid()
        # plt.ylabel('Total wealth')
        # plt.xlabel('Date')
        # plt.title('OLPS')
        # plt.show()



if __name__ == "__main__":
    #run = RUN(df,'UP')
    pass
