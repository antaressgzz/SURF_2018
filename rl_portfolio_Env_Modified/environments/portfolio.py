import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint
import logging
import os
import tempfile
import time
import gym
import gym.spaces

from ..config import eps
from ..data.utils import normalize, random_shift, scale_to_start
from ..util import MDD as max_drawdown, sharpe, talib_features
from ..callbacks.LivePlot import LivePlot
from ..callbacks.notebook_plot import LivePlotNotebook

logger = logging.getLogger(__name__)


class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, df, steps=252, input='price', trade_period=1, norm=None,
                 talib=False, scale_extra_cols=True,
                 augment=0.00, window_length=50, random_reset=True):
        """
        DataSrc.
        df - csv for data frame index of timestamps
             and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close',...]]
             an example is included as an hdf file in this repository
        steps - total steps in episode
        scale - scale the data for each episode
        scale_extra_cols - scale extra columns by global mean and std
        augment - fraction to augment the data by
        random_reset - reset to a random time (otherwise continue through time)
        """
        self.steps = steps
        self.augment = augment
        self.random_reset = random_reset
        self.trade_period = trade_period
        self.input = input
        self.norm = norm
        self.talib = talib
        self.scale_extra_cols = scale_extra_cols
        self.window_length = window_length
        self.idx = self.window_length + 1
        self.df = df.copy()

        self.reboot = True

        # get rid of NaN's
        df = df.copy()
        df.replace(np.nan, 0, inplace=True)
        df = df.fillna(method="pad")

        # dataframe to matrix
        self.asset_names = df.columns.levels[0].tolist()

        # print(self.asset_names)
        self.features = df.columns.levels[1].tolist()
        print('features:', self.features)
        data = df.as_matrix().reshape(
            (len(df), len(self.asset_names), len(self.features)))
        self.price_columns = self.features[:3]
        print(self.price_columns)
        self.nb_pc = len(self.price_columns)
        self.close_pos = self.price_columns.index('close')
        print('close_pos:', self.close_pos)
        self._data = np.transpose(data, (1, 0, 2))

        self.mean = self._data.mean(1)
        self.std = self._data.std(1)

        if len(self.features) == 4:
            self.vol_pos = 3
            self._data[:, :, self.vol_pos] -= self.mean[:, np.newaxis, self.vol_pos]
            self._data[:, :, self.vol_pos] /= self.mean[:, np.newaxis, self.vol_pos]

        if self.talib == False:
            print('data:', self._data.shape)
            self._times = df.index
        else:
            talibs_data = talib_features(self._data[:, :, self.close_pos].copy())
            mean = talibs_data.mean(1)
            std = talibs_data.std(1)
            talibs_data -= mean[:, np.newaxis, :]
            talibs_data /= std[:, np.newaxis, :]
            self._data = np.concatenate([self._data[:, 50:, :], talibs_data], axis=2)
            print('data:', self._data.shape)
            self._times = df.index[50:]

    def _step(self):
        # get history matrix from dataframe
        data_window = self.data[:, self.step*self.trade_period+1:self.step*self.trade_period+1+
                                             self.window_length].copy()


        # (eq.1) prices
        y1 = data_window[:, -1, self.close_pos] / data_window[:, -self.trade_period-1, self.close_pos]
        y1 = np.concatenate([[1.0], y1])  # add cash price

        # (eq 18) X: prices are divided by close price

        if self.input == 'price':
            if self.norm == 'latest_close':
                last_close_price = data_window[:, -1, self.close_pos]
                data_window[:, :, :self.nb_pc] /= last_close_price[:, np.newaxis, np.newaxis]
            elif self.norm == 'previous':
                _data_window = self.data[:,
                               self.step * self.trade_period:self.step * self.trade_period + self.window_length,
                               :self.nb_pc].copy()
                data_window[:, :, :self.nb_pc] /= _data_window
            elif self.norm == None:
                pass
            else:
                print('Invalid norm.')
        elif self.input == 'rf':
            _data_window = self.data[:,
                           self.step*self.trade_period:self.step*self.trade_period+self.window_length, :self.nb_pc].copy()
            data_window[:, :, :self.nb_pc] = data_window[:, :, :self.nb_pc] / _data_window - 1
            data_window[:, :, :self.nb_pc] *= 1000

        self.step += 1
        history = data_window
        done = bool(self.step >= self.steps)

        return history, y1, done

    def reset(self):
        self.step = 0
        # get data for this episode
        if not self.reboot:

            if self.random_reset:
                self.idx = np.random.randint(
                    low=self.window_length + 1, high=self._data.shape[1] - self.steps*self.trade_period - 2)
            else:
                # continue sequentially, before reseting to start
                if self.idx > (self._data.shape[1] - 2*self.steps*self.trade_period - 2):
                    self.idx = self.window_length + 1
                else:
                    self.idx += self.steps * self.trade_period
        else:
            self.reboot = False

        self.data = self._data[:, self.idx -
                             self.window_length-1:self.idx+self.steps*self.trade_period+1].copy()

        self.times = self._times[self.idx -
                                 self.window_length-1:self.idx+self.steps*self.trade_period+1]
        # augment data to prevent overfitting
        for i in range(len(self.asset_names)):
            for j in range(len(self.price_columns)):
                self.data[i, :, j] += \
                np.random.normal(loc=0, scale=self.augment*self.mean[i, j], size=self.data[i, :, j].shape)

        if len(self.features) == 4:
            self.data[:, :, self.vol_pos] += \
                np.random.normal(loc=0, scale=1000*self.augment, size=self.data[:, :, self.vol_pos].shape)

    def OLPS_data(self):

        dfs = []
        for i in self.df.columns.levels[0]:
            dfs.append(self.df[i]['close'])
        OLPS_df = pd.concat(dfs, axis=1)
        OLPS_df = OLPS_df.as_matrix()
        OLPS_low = self.idx
        OLPS_high = self.idx + self.steps*self.trade_period
        idxs = np.linspace(OLPS_low, OLPS_high, num=self.steps, endpoint=False).astype(int)
        return OLPS_df[idxs]


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=[], steps=128, trading_cost=0.0025, time_cost=0.0):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        # self.reset()

    def _step(self, w0, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        # w0 = self.w0
        # p0 = self.p0

        _p0 = self._p0
        _w0 = self._w0
        if w0 is None:
            w0 = _w0
        c1 = self.cost * (np.abs(_w0[1:]-w0[1:])).sum()
        w0_ = (y1 * w0) / np.dot(y1, w0)
        p0_ = _p0 * (1 - c1) * np.dot(w0, y1)

        # dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into
        # (eq16) cost to change portfolio
        # (excluding change in cash to avoid double counting for transaction cost)
        # c1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum()
        # p1 = p0 * (1 - c1) * np.dot(y1, w0)  # (eq11) final portfolio value
        # p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding
        # can't have negative holdings in this model (no shorts)
        # p1 = np.clip(p1, 0, np.inf)
        # rho1 = p1 / p0 - 1  # rate of returns
        rho1 = p0_ / _p0 - 1  # rate of returns

        # r1 = np.log((p1 + eps) / (p0 + eps))  # (eq10) log rate of return
        r1 = np.log(p0_ / _p0)

        # (eq22) immediate reward is log rate of return scaled by episode length
        # reward = r1 / self.steps
        reward = r1

        # remember for next step
        self._w0 = w0_
        self._p0 = p0_

        # self.w0 = w1
        # self.p0 = p1
        # if we run out of money, we're done
        done = bool(p0_ == 0)

        # should only return single values, not list
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p0_,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights": w0,
            "weights_mean": w0.mean(),
            "weights_std": w0.std(),
            "cost": c1,
        }

        # record weights and prices
        for i, name in enumerate(['USD_USD'] + self.asset_names):
            info['weight_' + name] = w0[i]
            info['price_' + name] = y1[i]

        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self._w0 = np.array([1.0] + [0.0] * len(self.asset_names))
        self._p0 = 1.0


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['notebook', 'ansi', 'usual']}

    def __init__(self,
                 df,
                 steps=256,
                 trading_cost=0.0025,
                 trade_period=1,
                 time_cost=0.00,
                 window_length=50,
                 talib=False,
                 input='price',
                 norm=None,
                 augment=0.00,
                 output_mode='EIIE',
                 log_dir=None,
                 scale_extra_cols=True,
                 random_reset=True
                 ):
        """
        An environment for financial portfolio management.
        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
            steps - steps in episode
            window_length - how many past observations["history"] to return
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            augment - fraction to randomly shift data by
            output_mode: decides observation["history"] shape
            - 'EIIE' for (assets, window, 3)
            - 'atari' for (window, window, 3) (assets is padded)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """
        self.src = DataSrc(df=df, steps=steps, input=input, trade_period=trade_period, talib=talib,
                           norm=norm, scale_extra_cols=scale_extra_cols, augment=augment, window_length=window_length,
                           random_reset=random_reset)
        self._plot = self._plot2 = self._plot3 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)
        self.log_dir = log_dir

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        # cash bias become the actual corresponding portfolio weights
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=nb_assets + 1)

        # get the history space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (
                nb_assets,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'atari':
            obs_shape = (
                window_length,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'mlp':
            obs_shape = (nb_assets) * window_length * \
                        (len(self.src.features))
        else:
            raise Exception('Invalid value for output_mode: %s' %
                            self.output_mode)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                -10,
                10,
                obs_shape
            ),
            'weights': self.action_space
        })
        # self._reset()

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """

        if action is not None:
            logger.debug('action: %s', action)
            weights = np.clip(action, 0.0, 1.0)
            weights /= weights.sum()
            # Sanity checks
            assert self.action_space.contains(
                action), 'action should be within %r but is %r' % (self.action_space, action)
            np.testing.assert_almost_equal(
                np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)
        else:
            weights = action

        history, y1, done1 = self.src._step()

        # print('jistory',history[:, -10:, 2])
        # print('y1:', y1)
        # print('---------------------')

        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod(
            [inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.src.times[self.src.step*self.src.trade_period+self.src.window_length+1].timestamp()
        info['steps'] = self.src.step

        self.infos.append(info)

        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'atari':
            padding = history.shape[1] - history.shape[0]
            history = np.pad(history, [[0, padding], [
                0, 0], [0, 0]], mode='constant')
        elif self.output_mode == 'mlp':
            history = history.flatten()

        return {'history': history, 'weights': info["weights"]}, reward, done1 or done2, info

    def _reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        # action = self.sim.w0
        action = self.sim._w0
        observation, reward, done, info = self.step(action)
        return observation

    def _seed(self, seed):
        np.random.seed(seed)
        return [seed]

    def _render(self, mode='usual', close=False):
        # if close:
        # return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'notebook':
            self.plot(LivePlotNotebook, mode, close)
        elif mode == 'usual':
            self.plot(LivePlot, mode, close)

    #       self.plot_notebook(close)

    def plot(self, tool, mode, close=False):
        """Live plot using the jupyter notebook rendering of matplotlib."""

        if close:
            self._plot = self._plot2 = self._plot3 = None
            return

        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')

        # plot prices and performance
        all_assets = ['USD_USD'] + self.sim.asset_names
        if not self._plot:
            colors = [None] * len(all_assets) + ['black']
            self._plot_dir = os.path.join(
                self.log_dir, '_plot_prices_' + str(time.time())) if self.log_dir else None
            self._plot = tool(
                log_dir=self._plot_dir, title='prices & performance', labels=all_assets + ["Portfolio"], ylabel='value',
                colors=colors)
        y_return = df_info["market_return"]
        x = df_info.index
        y_portfolio = df_info["portfolio_value"]
        y_assets = [df_info['price_' + name].cumprod()
                    for name in all_assets]
        self._plot.update(x, y_assets + [y_portfolio])
        self.maxDD = round(max_drawdown(y_portfolio) * 100, 2)
        self.sharp = round(sharpe(y_portfolio) * 100, 2)
        print('Max Draw Down: ', self.maxDD)
        print('Sharp Ratio: ', self.sharp)

        if mode == 'usual':
            self._plot.end()

        # plot portfolio weights
        if not self._plot2:
            self._plot_dir2 = os.path.join(
                self.log_dir, '_plot_weights_' + str(time.time())) if self.log_dir else None
            self._plot2 = tool(
                log_dir=self._plot_dir2, labels=all_assets, title='weights', ylabel='weight')
        ys = [df_info['weight_' + name] for name in all_assets]
        self._plot2.update(x, ys)
        # if mode == 'usual':
        #     self._plot2.end()

        # plot portfolio costs
        if not self._plot3:
            self._plot_dir3 = os.path.join(
                self.log_dir, '_plot_cost_' + str(time.time())) if self.log_dir else None
            self._plot3 = tool(
                log_dir=self._plot_dir3, labels=['cost'], title='costs', ylabel='cost')
        ys = [df_info['cost'].cumsum()]
        self._plot3.update(x, ys)
        if mode == 'usual':
            self._plot3.end()

        if close:
            self._plot = self._plot2 = self._plot3 = None

    def return_df(self):

        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')
        return df_info

    def return_SD(self):
        return self.sharp, self.maxDD

    def OLPS_data(self):
        return self.src.OLPS_data()