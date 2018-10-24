import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
from hypertune import start_commander, start_workers
from coordinator import Coordinator
from config import mode, number_workers, tuned_config
from hypertune import construct_config
# %matplotlib inline


if mode == 'parallel':
    start_commander()
    workers = start_workers(number_workers)
else:
    model = Coordinator(tuned_config, '-5.28_fee')
    # model.restore('2102-160000')
    model.train('single', True)
    model.back_test('test', 12000, True)