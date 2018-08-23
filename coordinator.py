import sys
import os
current_path = os.path.abspath(__file__)
OLPS_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + 'OLPS/OLPS_modified')
sys.path.append(OLPS_path)
from OLPS.olps import OLPS
import numpy as np


class Coordinator:
    def __init__(self, agent, trade_period):
        self.agent = agent
        self.trade_period = trade_period

    def train(self, env, total_training_step, replay_period, tensorboard=False):

        self.env = env
        if tensorboard == True:
            self.agent.initialize_tb()
        else:
            self.agent.tensorboard = False

        self.total_training_step = total_training_step
        self.replay_period = replay_period

        training_step = 0
        while training_step < self.total_training_step:
            observation = self.env.reset()
            action_idx, action = self.agent.choose_action(observation)
            while True:
                if (training_step + 1) % self.trade_period == 0:
                    action_idx, action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                reward *= 100
                reward = np.clip(reward, -1, 1)
                self.agent.store(observation, action_idx, reward, observation_)
                observation = observation_

                if self.agent.start_replay():
                    if self.agent.memory_cnt() % self.replay_period == 0:
                        self.agent.replay()  # update target
                        training_step = self.agent.get_training_step()
                        if (training_step - 1) % 5000 == 0:
                            print('training_step: {}, epsilon: {:.2f}, ave_reward: {:.2e}'.format(
                            training_step, self.agent.epsilon, self.agent.get_ave_reward()/100))
                if done:
                    break

        print("Successfully trained.")

    def back_test(self, env_test, render_mode='usual'):
        test_step = 0
        ob = env_test.reset()
        action_idx, action = self.agent.choose_action(ob, test=True)
        while True:
            if (test_step + 1) % self.trade_period == 0:
                action_idx, action = self.agent.choose_action(ob, test=True)
            ob, reward, done, _ = env_test.step(action)
            test_step += 1
            if done:
                break

        df_info = env_test.return_df()
        df_olps = env_test.OLPS_data()
        olps = OLPS(df_olps=df_olps, df_info=df_info, algo="OLMAR,ONS")
        olps.plot()
        env_test.render(render_mode)
        sharp, maxDD = env_test.return_SD()
        C_Sharp, C_MDD = olps.IndexComparison(sharp, maxDD)
        print('Sharp ratio Comparison: ', C_Sharp, ' % \nMDD Comparison: ', C_MDD, ' %')
        print('Test is end.')

    def open_tb(self, port):
        path = os.path.abspath('logs/train/'+self.agent.name+' --port='+port)
        os.system('tensorboard --logdir='+path)

    def restore(self, name):
        self.agent.restore(name)

    def network_state(self):
        return  self.agent.network_state()

    def action_values(self, o):
        return self.agent.action_values(o)