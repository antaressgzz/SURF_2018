import sys
import os
current_path = os.path.abspath(__file__)
OLPS_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + 'OLPS/OLPS_modified')
sys.path.append(OLPS_path)
from OLPS.olps import OLPS
import numpy as np
import matplotlib.pyplot as plt

class Coordinator:
    def __init__(self, agent):
        self.agent = agent

    def train(self, env_train, env_val, total_training_step, replay_period, tensorboard=False):

        self.env_train = env_train
        if tensorboard == True:
            self.agent.initialize_tb()
        else:
            self.agent.tensorboard = False

        self.total_training_step = total_training_step
        self.replay_period = replay_period
        training_step = 0
        self.rewards = []
        self.ave_rewards = []

        def get_val_reward():
            ob = env_val.reset()
            val_rewards = []
            while True:
                action_idx, action = self.agent.choose_action(ob, test=True)
                ob, reward, done, _ = env_val.step(action)
                val_rewards.append(reward)
                if done:
                    break
            return np.mean(val_rewards)

        while training_step < self.total_training_step:
            observation = self.env_train.reset()
            while True:
                action_idx, action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env_train.step(action)
                self.rewards.append(reward)
                reward *= 1000
                # reward = np.clip(reward, -1, 1)
                self.agent.store(observation, action_idx, reward, observation_)
                # print('-----------------')
                # print('observation', observation['history'][0, 1:10, :])
                # print('observation_', observation_['history'][0, 1:10, :])

                observation = observation_
                if self.agent.start_replay():
                    # if training_step % 500 == 0:
                    #     print(reward)
                    if self.agent.memory_cnt() % self.replay_period == 0:
                        self.agent.replay()  # update target
                        training_step = self.agent.get_training_step()
                        if (training_step - 1) % 5000 == 0:
                            num_r = 50000
                            train_r = np.sum(self.rewards[-num_r:]) / num_r
                            self.ave_rewards.append(train_r)
                            val_r = get_val_reward()
                            print('training_step: {}, epsilon: {:.2f}, train_r: {:.2e}, val_r:{:.2e}'.format(
                            training_step, self.agent.epsilon, train_r, val_r))

                if done:
                    break

        print("Successfully trained.")
        x = np.arange(len(self.ave_rewards))
        fig, ax = plt.subplots()
        ax.plot(x, self.ave_rewards)
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.show()

    def back_test(self, env_test, render_mode='usual'):
        ob = env_test.reset()
#         print(ob['history'][0, -10:, 2])
#         print(self.agent.action_values(ob))
        rewards = 0
        while True:
            action_idx, action = self.agent.choose_action(ob, test=True)
            ob, reward, done, _ = env_test.step(action)
            rewards += reward
            if done:
                break

        print('total rewards:', rewards)
        df_info = env_test.return_df()
        df_olps = env_test.OLPS_data()
        olps = OLPS(df_olps=df_olps, df_info=df_info, algo="BAH,BestSoFar")
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