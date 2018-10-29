import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
from hypertune import start_commander, start_workers
from coordinator import Coordinator
from config import mode, number_workers, tuned_config, fix_random_seed
import numpy as np

if fix_random_seed:
    np.random.seed(123)

# %matplotlib inline

if mode == 'parallel':
    start_commander()
    workers = start_workers(number_workers)
else:
    model = Coordinator(tuned_config, '2900')
    #################### to do ####################
    # model = Coordinator(tuned_config, '-5.28')
    # model.restore_price_predictor('-5.28-80000-')
    ##############################################
    model.train('single', True)
    model.back_test('test', 2500, True)