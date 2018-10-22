from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from coordinator import Coordinator
import numpy as np
import time
import multiprocessing as mp
import shlex
from subprocess import Popen
import logging
import pprint
from config import get_config
from math import log
import sys
import json
from config import FEE, EXP_KEY, MAX_EVALS

losses = []
f = open('./logs/' + str(EXP_KEY) + '.json', 'a')

def loguniform(name, low, high):
    return hp.loguniform(name, log(low), log(high))

def qloguniform(name, low, high, q):
    return hp.qloguniform(name, log(low), log(high), q)

# Search space
param_space = {
    # train
    'steps': hp.quniform("steps", 100000, 240000, 40000),
    'learning_rate': loguniform('learning_rate', 1e-5, 1e-3),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'replay_period': hp.choice('replay_period', [2, 4, 8, 16]),
    'division': hp.choice('division', [3, 4, 5, 6]),
    'dropout': hp.uniform('dropout', 0.3, 0.8),
    'reward_scale': hp.quniform('reward_scale', 100, 2100, 200),
    #net
    'cnn_activation': hp.choice('cnn_activation', ['selu', 'relu', 'leaky_relu']),
    'fc_activation': hp.choice('fc_activation', ['selu', 'relu', 'leaky_relu']),
    'fc_size': hp.choice('fc_size', [32, 64, 128, 256]),
    'kernels': hp.choice('kernels',
            [[[1, hp.quniform('k_w11', 3, 10, 1)], [4, hp.quniform('k_w12', 3, 10, 1)]],
            [[4, hp.quniform('k_w21', 3, 10, 1)], [1, hp.quniform('k_w22', 3, 10, 1)]],
            [[1, hp.quniform('k_w31', 3, 10, 1)], [1, hp.quniform('k_w32', 3, 10, 1)]]]),
    'filters': [hp.quniform('filter1', 2, 10, 1), hp.quniform('filter2', 2, 10, 1)],
    'strides': [[1, hp.quniform('strides1', 1, 3, 1)],
               [1, hp.quniform('strides2', 1, 3, 1)]],
    'regularizer': loguniform('weight_decay', 1e-5, 1e-2),
    # 'padding': hp.choice('padding', ['same', 'valid']),
    # env
    'window_length': hp.quniform('window_length', 50, 300, 50),
    'input': hp.choice('input', ['rf', 'price']),
    'norm': hp.choice('norm', ['latest_close', 'previous']),
}

param_space_fee = {
    # train
    'steps': hp.quniform("steps", 100000, 280000, 40000),
    'learning_rate': loguniform('learning_rate', 1e-5, 1e-3),
    'process_cost': hp.choice('process_cost', [True, False]),
    'discount': 1 - loguniform('discount', 1e-4, 1),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'replay_period': hp.choice('replay_period', [2, 4, 8, 16]),
    'division': hp.choice('division', [3, 4, 5, 6]),
    'dropout': hp.uniform('dropout', 0.3, 0.8),
    'reward_scale': hp.quniform('reward_scale', 100, 2100, 200),
    #net
    'activation': hp.choice('activation', ['selu', 'relu', 'leaky_relu']),
    'fc1_size': hp.choice('fc_size', [32, 64, 128, 256]),
    'fc2_size': hp.choice('fc2_size', [32, 64, 128, 256]),
    'upd_tar_prd': hp.quniform("upd_tar_prd", 1000, 10000, 1000),
    'kernels': hp.choice('kernels',
            [[[1, hp.quniform('k_w11', 3, 10, 1)], [4, hp.quniform('k_w12', 3, 10, 1)]],
            [[4, hp.quniform('k_w21', 3, 10, 1)], [1, hp.quniform('k_w22', 3, 10, 1)]],
            [[1, hp.quniform('k_w31', 3, 10, 1)], [1, hp.quniform('k_w32', 3, 10, 1)]]]),
    'filters': [hp.quniform('filter1', 2, 10, 1), hp.quniform('filter2', 2, 10, 1)],
    'strides': [[1, hp.quniform('strides1', 1, 3, 1)],
               [1, hp.quniform('strides2', 1, 3, 1)]],
    'regularizer': loguniform('weight_decay', 1e-5, 1e-2),
    # 'padding': hp.choice('padding', ['same', 'valid']),
    'cnn_activation': hp.choice('cnn_activation', ['selu', 'relu', 'leaky_relu']),
    'fc_activation': hp.choice('fc_activation', ['selu', 'relu', 'leaky_relu']),
    # env
    'window_length': hp.quniform('window_length', 50, 300, 50),
    'input': hp.choice('input', ['rf', 'price']),
    'norm': hp.choice('norm', ['latest_close', 'previous']),
}


def construct_config(config, para):
    trainc = config["train"]
    netc = config["net"]
    envc = config["env"]
    # train
    print(trainc)
    print(para)
    trainc["steps"] = int(para["steps"])
    # trainc["steps"] = 1001
    trainc["learning_rate"] = para["learning_rate"]
    trainc["division"] = int(para["division"])
    # trainc["upd_tar_prd"] = int(para["upd_tar_prd"])
    trainc["batch_size"] = int(para["batch_size"])
    trainc["replay_period"] = int(para['replay_period'])
    # trainc["discount"] = para["discount"]
    trainc['dropout'] = para['dropout']
    trainc['reward_scale'] = para['reward_scale']
    # env
    envc["window_length"] = int(para["window_length"])
    envc['input'] = para['input']
    envc['norm'] = para['norm']
    # net
    netc['cnn_activation'] = para['cnn_activation']
    netc['fc_activation'] = para['fc_activation']
    if FEE == True:
        netc['fc2_size'] = para['fc2_size']
        trainc['process_cost'] = para['process_cost']
        trainc['discount'] = para['discount']
    else:
        netc['fc_size'] = para['fc_size']
    netc['kernels'] = [list(map(int, k)) for k in para['kernels']]
    # netc['padding'] = para['padding']
    netc['filters'] = list(map(int, para['filters']))
    netc['strides'] = [list(map(int, k)) for k in para['strides']]
    netc['regularizer'] = para['regularizer']
    return config


# ob func
def train_one(tuning_params):
    start = time.time()
    config = get_config(FEE)
    config = construct_config(config, tuning_params)
    coo = Coordinator(config, str(EXP_KEY))
    val_rewards, tr_rs = coo.evaluate()
    loss = -1 * np.mean(val_rewards[-6:]) * 1e6
    eval_time = time.time() - start
    log_training(config, val_rewards, tr_rs, loss, eval_time)
    result = {
              'loss': loss,
              'status': STATUS_OK,
              'val': val_rewards,
              'train': tr_rs,
              }
    return result

def start_server():
    trials = MongoTrials('mongo://localhost:27017'
                         '/hyperopt/jobs', exp_key=EXP_KEY)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    if FEE:
        params = param_space_fee
    else:
        params = param_space
    best = fmin(
         train_one,
         trials=trials,
         space=params,
         algo=tpe.suggest,
         max_evals=MAX_EVALS,
    )
    pprint.pprint(best)
    # pprint.pprint(trials.trials)

def start_commander():
    mp.Process(target=start_server).start()

def worker(index):
    cmd = "hyperopt-mongo-worker --mongo=localhost:27017/hyperopt"
    args = shlex.split(cmd)
    process = Popen(args)
    return process

def start_workers(processes=2):
    p_num = int(processes)
    p = mp.Pool(processes=p_num)
    logging.info("The remaining cpu amount is {}".format(mp.cpu_count()))
    pses = p.map(worker, range(0,p_num))
    return pses

# Write the results in a txt file
def log_training(config, val_rs, tr_rs, loss, eval_time):
    losses.append(loss)
    js_dic = json.dumps(config)
    f.write(js_dic+'\n')
    f.writelines('val_rewards:'+str(val_rs)+'\n')
    f.writelines('tra_rewards:'+str(tr_rs)+'\n')
    f.writelines('loss:'+str(loss)+'\n')
    f.writelines('eval_time:'+str(eval_time)+'\n')
    f.writelines('\n')