import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
from hypertune import start_commander, start_workers
from coordinator import Coordinator
from config import get_config, param_space_fee, param_space, FEE, EXP_KEY
from hypertune import construct_config
# %matplotlib inline

mode = 'single'
name = '2200'

if mode == 'auto':
    start_commander()
    workers = start_workers(4)
else:
    config = get_config(False)
    if FEE:
        params = param_space_fee
    else:
        params = param_space
    config = construct_config(config, params)
    model = Coordinator(config, name)
    # model.restore('2102-160000')
    model.train('single', True)
    model.back_test('test', 12000, True)