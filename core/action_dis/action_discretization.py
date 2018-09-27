from itertools import combinations
from scipy.special import comb
import numpy as np
import pprint

def action_discretization(asset_num, division):
    item_num = asset_num + division - 1
    action_num = int(comb(item_num, asset_num - 1))
    actions = {}
    pointer = 0
    for c in combinations(np.arange(item_num), asset_num - 1):
        action = np.zeros(asset_num)
        for i in range(len(c) - 1):
            action[i + 1] = c[i + 1] - c[i] - 1
        action[0] = c[0]
        action[-1] = item_num - c[-1] - 1
        actions[pointer] = action / division
        pointer += 1
    # One additional action
    # actions[action_num] = None
    # action_num += 1
    return action_num, actions

def test(asset_num, division):
    action_num, actions = action_discretization(asset_num, division)
    assert action_num == len(actions)
    pp = pprint.PrettyPrinter(indent=0)
    print('actions dict:')
    pp.pprint(actions)
    print('action number: ', action_num)
    for i in range(action_num-1):
        assert np.sum(actions[i]) == 1

if __name__ == '__main__':
    test(5, 5)