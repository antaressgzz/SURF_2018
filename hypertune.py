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

EXP_KEY = 1700
MAX_EVALS = 24
FEE = False

def loguniform(name, low, high):
    return hp.loguniform(name, log(low), log(high))

def qloguniform(name, low, high, q):
    return hp.qloguniform(name, log(low), log(high), q)

# Search space
param_space = {
    # train
    'steps': hp.quniform("steps", 60000, 120000, 20000),
    'learning_rate': loguniform('learning_rate', 1e-5, 1e-3),
    # 'discount': hp.uniform('discount', 0, 1),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'replay_period': hp.choice('replay_period', [2, 4, 8, 16]),
    'division': hp.choice('division', [3, 4, 5, 6]),
    'dropout': hp.uniform('dropout', 0.3, 0.8),
    #net
    'activation': hp.choice('activation', ['selu', 'relu', 'leaky_relu']),
    'fc_size': hp.choice('fc_size', [32, 64, 128, 256]),
    'kernel': [[hp.choice('k_width1', [1, 4]), hp.quniform('k_h1', 3, 10, 1)],      # this needs changing if the assets number changes.
               [hp.choice('k_width2', [1, 4]), hp.quniform('k_h2', 3, 10, 1)]],
    'filter': [hp.quniform('filter1', 2, 10, 1), hp.quniform('filter2', 2, 10, 1)],
    'strides': [hp.quniform('strides1', 1, 5, 1), hp.quniform('strides2', 1, 5, 1)],
    'regularizer': loguniform('weight_decay', 1e-5, 1e-2),
    # env
    'window_length': hp.quniform('window_length', 50, 300, 50),
    'input': hp.choice('input', ['rf', 'price']),
    'norm': hp.choice('norm', ['latest_close', 'previous']),
    'argument': loguniform('argument', 1e-6, 1e-3)
}

def construct_config(config, para):
    trainc = config["train"]
    netc = config["net"]
    envc = config["env"]
    # train
    trainc["steps"] = int(para["steps"])
    trainc["learning_rate"] = para["learning_rate"]
    trainc["division"] = int(para["division"])
    trainc["batch_size"] = int(para["batch_size"])
    trainc["replay_period"] = int(para['replay_period'])
    # trainc["discount"] = para["discount"]
    trainc['dropout'] = para['dropout']
    # env
    envc["window_length"] = int(para["window_length"])
    envc['input'] = para['input']
    envc['norm'] = para['norm']
    envc['argument'] = para['argument']
    # net
    netc['activaton'] = para['activation']
    netc['fc_size'] = para['fc_size']
    netc['kernel'] = [list(map(int, k)) for k in para['kernel']]
    netc['filter'] = list(map(int, para['filter']))
    netc['strides'] = list(map(int, para['strides']))
    netc['regularizer'] = para['regularizer']
    return config


# ob func
def train_one(tuning_params):
    start = time.time()
    config = get_config(FEE)
    config = construct_config(config, tuning_params)
    print('start_new')
    coo = Coordinator(config, str(EXP_KEY))
    val_rewards, tr_rs = coo.evaluate()
    loss = -1 * np.mean(val_rewards[-4:]) * 1e6
    result = {
              'loss': loss,
              'status': STATUS_OK,
              'val': val_rewards,
              'train': tr_rs,
              'eval_time': time.time() - start
              }
    return result

def start_server():
    trials = MongoTrials('mongo://localhost:27017'
                         '/hyperopt/jobs', exp_key=EXP_KEY)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    best = fmin(
         train_one,
         trials=trials,
         space=param_space,
         algo=tpe.suggest,
         max_evals=MAX_EVALS,
    )
    pprint.pprint(best)
    pprint.pprint(trials.trials)
    log_training(trials.trials, best)

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

def arrange_cpu():
    pass

# Write the results in a txt file
def log_training(trials, best):
    f = open('./logs/'+time.strftime("%Y%m%d%H%M", time.localtime())+'.txt', 'w')
    losses = []
    for i in range(MAX_EVALS):
        dic = trials[i]
        f.writelines('model：'+str(i)+'\n')
        losses.append(dic['result']['loss'])
        f.writelines('exp_key: '+str(dic['exp_key'])+'\n')
        f.writelines('params: '+str(dic['misc']['vals'])+'\n')
        f.writelines('result: '+str(dic['result'])+'\n')
        f.writelines('\n')
    f.writelines('Best model:'+str(np.argmin(losses))+' Params:'+str(best))
    f.close()