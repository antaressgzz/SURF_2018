from universal.RUN import RUN
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class OLPS():
    def __init__(self, df_olps, df_info, algo):
        run = RUN(df_olps, algo)
        run.AlgosCalcu()
        self.res, self.options, self.sharpCollect, self.MDDCollect = run.return_df()
        self.df_info = df_info

    def plot(self):
        x = self.df_info.index
        y = self.df_info["portfolio_value"]
        for i in range(len(self.res)):
            plt.plot(x, self.res[i])
        plt.plot(x, y)
        self.options.append('dqn')
        plt.legend(self.options)
        plt.grid()
        plt.ylabel('Total wealth')
        plt.xlabel('Date')
        plt.title('Comparison with OLPS')
        plt.show()

    def IndexComparison(self, sharp, maxDD):
        OLPS_Sharp = max(self.sharpCollect) * 100
        OLPS_MDD = min(self.MDDCollect) * 100
        DQN_Sharp = sharp
        DQN_MDD = maxDD
        C_Sharp = round((DQN_Sharp - OLPS_Sharp), 2)
        C_MDD = round((DQN_MDD - OLPS_MDD), 2)

        return C_Sharp, C_MDD