import numpy as np
from core.action_dis.action_discretization import action_discretization


class Memory:
    def __init__(self, action_num, actions ,memory_size=10000, batch_size=16):
        self.memory_dict = {}
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_pointer = 0
        self.action_num=action_num
        self.actions=actions
        self.rewards = []
        self.full = False

    def sample(self):
        observations = {'history':np.zeros((self.batch_size, self.history_size[0], self.history_size[1], self.history_size[2])),
                        'weights':np.zeros((self.batch_size, self.weight_size[0]))}
        observations_ = {'history':np.zeros((self.batch_size, self.history_size[0], self.history_size[1], self.history_size[2])),
                         'weights':np.zeros((self.batch_size, self.weight_size[0]))}

        # observations = {}
        # observations_ = {}

        actions_idx = []
        rewards = []

        batch_idx = np.random.choice(self.memory_size, self.batch_size, replace=False)

        for i, idx in enumerate(batch_idx):
            observations['history'][i, :, :, :] = self.memory_dict[idx][0]['history']
            observations['weights'][i, :] = self.memory_dict[idx][0]['weights']
            observations_['history'][i, :, :, :] = self.memory_dict[idx][3]['history']
            observations_['weights'][i, :] = self.memory_dict[idx][3]['weights']
            # print(self.actions,'\n',self.memory_dict[idx][1])
            # print(np.where(self.actions==np.array(self.memory_dict[idx][1])))
            # actions.append(np.where(self.actions == np.array(self.memory_dict[idx][1])))
            actions_idx.append(self.memory_dict[idx][1])
            rewards.append(self.memory_dict[idx][2])

        # observations['history'] = np.stack([self.memory_dict[idx][0]['history'] for idx in batch_idx], axis=0)
        # observations['weights'] = np.stack([self.memory_dict[idx][0]['weights'] for idx in batch_idx], axis=0)
        # observations_['history'] = np.stack([self.memory_dict[idx][3]['history'] for idx in batch_idx], axis=0)
        # observations_['weights'] = np.stack([self.memory_dict[idx][3]['weights'] for idx in batch_idx], axis=0)
        # actions_idx = np.array([self.memory_dict[idx][1] for idx in batch_idx])
        # rewards = np.array([self.memory_dict[idx][1] for idx in batch_idx])

        ############# Test ##################
        # assert (observations['history'] != 0).all()
        # assert np.isclose(observations_['weights'][0], self.actions[actions_idx[0]]).all()
        # assert (observations_['weights'][0] == self.actions[actions_idx[0]]).all()
        return observations, actions_idx, rewards, observations_

    def store(self, observation, action, reward, observation_):
        if self.memory_pointer == self.memory_size:
            self.memory_pointer -= self.memory_size
            self.full = True
            self.history_size = np.shape(self.memory_dict[0][0]['history'])
            self.weight_size = np.shape(self.memory_dict[0][0]['weights'])
            # print('Memory is full, pointer is reset.')
        self.memory_dict[self.memory_pointer] = (observation, action, reward, observation_)
        self.memory_pointer += 1
        self.rewards.append(reward)

    def get_ave_reward(self):
        rewards = np.array(self.rewards)
        self.rewards = []
        return np.mean(rewards)

    def start_replay(self):
        return self.full

if __name__ == '__main__':
    import pandas as pd
    from rl_portfolio_Env_Modified.environments import PortfolioEnv

    df_train = pd.read_hdf('./data/forex_30m_CLOSE_1.hf', key='train')
    df_test = pd.read_hdf('./data/forex_30m_CLOSE_1.hf', key='test')

    env = PortfolioEnv(df_train,
                       steps=2000,
                       trading_cost=0.00007,
                       time_cost=0.00,
                       window_length=50)
    asset_num = 5
    division = 1
    action_num, actions = action_discretization(asset_num, division)

    training_step = 0
    memory_counter = 0
    memory = Memory(action_num, actions, memory_size=1000)
    observation = env.reset()
    while True:
        action_idx = np.random.randint(0, action_num)
        action = actions[action_idx]
        observation_, reward, done, info = env.step(action)
        memory.store(observation, action_idx, reward, observation_)
        if memory.full:
            o, a, r, o_ = memory.sample()
        if done:
            break