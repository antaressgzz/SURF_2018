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
from config import get_config, tuned_config
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
    'steps': hp.quniform("steps", 60000, 180000, 30000),
    'learning_rate': loguniform('learning_rate', 1e-5, 1e-3),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'replay_period': hp.choice('replay_period', [2, 4, 8, 16]),
    'division': hp.choice('division', [3, 4, 5, 6]),
    'dropout': hp.uniform('dropout', 0.3, 0.8),
    'reward_scale': hp.quniform('reward_scale', 100, 2100, 200),
    #net
    'cnn_activation': hp.choice('cnn_activation', ['selu', 'relu', 'leaky_relu']),
    'fc_activation': hp.choice('fc_activation', ['selu', 'relu', 'leaky_relu']),
    'fc1_size': hp.choice('fc1_size', [32, 64, 128, 256]),
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
    'trading_period': hp.choice('trading_period', [1, 2, 4, 8, 16, 32, 64])
}

param_space_fee = {
    # train
    # 'steps': hp.quniform("steps", 120000, 280000, 40000),
    'learning_rate': loguniform('learning_rate', 1e-5, 1e-3),
    'process_cost': hp.choice('process_cost', [True, False]),
    'discount': 1 - loguniform('discount', 1e-4, 1),
    # 'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    # 'replay_period': hp.choice('replay_period', [2, 4, 8, 16]),
    'division': hp.quniform('division', 4, 12, 1),
    # 'dropout': hp.uniform('dropout', 0.3, 0.8),
    'reward_scale': hp.quniform('reward_scale', 100, 2100, 200),
    'upd_tar_prd': hp.quniform("upd_tar_prd", 1000, 10000, 1000),
    #net
    # 'fc1_size': hp.choice('fc_size', [32, 64, 128, 256]),
    'fc2_size': hp.choice('fc2_size', [32, 64, 128, 256]),
    # 'kernels': hp.choice('kernels',
    #         [[[1, hp.quniform('k_w11', 3, 10, 1)], [4, hp.quniform('k_w12', 3, 10, 1)]],
    #         [[4, hp.quniform('k_w21', 3, 10, 1)], [1, hp.quniform('k_w22', 3, 10, 1)]],
    #         [[1, hp.quniform('k_w31', 3, 10, 1)], [1, hp.quniform('k_w32', 3, 10, 1)]]]),
    # 'filters': [hp.quniform('filter1', 2, 10, 1), hp.quniform('filter2', 2, 10, 1)],
    # 'strides': [[1, hp.quniform('strides1', 1, 3, 1)],
    #            [1, hp.quniform('strides2', 1, 3, 1)]],
    # 'regularizer': loguniform('weight_decay', 1e-5, 1e-2),
    # 'padding': hp.choice('padding', ['same', 'valid']),
    # 'cnn_activation': hp.choice('cnn_activation', ['selu', 'relu', 'leaky_relu']),
    # 'fc_activation': hp.choice('fc_activation', ['selu', 'relu', 'leaky_relu']),
    # env
    # 'window_length': hp.quniform('window_length', 50, 300, 50),
    # 'input': hp.choice('input', ['rf', 'price']),
    # 'norm': hp.choice('norm', ['latest_close', 'previous']),
    # 'trading_period': hp.quniform('trading_period', 1, 49, 4)
}


def construct_config(config, para):
    trainc = config["train"]
    netc = config["net"]
    envc = config["env"]
    # train
    print(trainc)
    print(para)
    try:
        trainc["steps"] = int(para["steps"])
    except:
        pass
    try:
        trainc["learning_rate"] = para["learning_rate"]
    except:
        pass
    try:
        trainc["division"] = int(para["division"])
    except:
        pass
    try:
        trainc["batch_size"] = int(para["batch_size"])
    except:
        pass
    try:
        trainc["replay_period"] = int(para['replay_period'])
    except:
        pass
    try:
        trainc['dropout'] = para['dropout']
    except:
        pass
    try:
        trainc['reward_scale'] = para['reward_scale']
    except:
        pass
    # env
    try:
        envc["window_length"] = int(para["window_length"])
    except:
        pass
    try:
        envc['trading_period']= int(para['trading_period'])
    except:
        pass
    try:
        envc['input'] = para['input']
    except:
        pass
    try:
        envc['norm'] = para['norm']
    except:
        pass
    # net
    try:
        netc['cnn_activation'] = para['cnn_activation']
    except:
        pass
    try:
        netc['fc_activation'] = para['fc_activation']
    except:
        pass
    try:
        netc['kernels'] = [list(map(int, k)) for k in para['kernels']]

    except:
        pass
    try:
        netc['filters'] = list(map(int, para['filters']))

    except:
        pass
    try:
        netc['strides'] = [list(map(int, k)) for k in para['strides']]

    except:
        pass
    try:
        netc['regularizer'] = para['regularizer']
    except:
        pass
    if FEE == True:
        netc['fc2_size'] = para['fc2_size']
        trainc['process_cost'] = para['process_cost']
        trainc['discount'] = para['discount']
        trainc["upd_tar_prd"] = int(para["upd_tar_prd"])
    else:
        netc['fc1_size'] = para['fc1_size']
    return config


# ob func
def train_one(tuning_params):
    start = time.time()
    if FEE:
        config = tuned_config
    else:
        config = get_config(FEE)
    config = construct_config(config, tuning_params)
    ################### To Do ################
    # coo = Coordinator(config, [name, ''])
    # coo.restore_price_predictor('-5.28-80000-')
    coo = Coordinator(config, 'search')
    ##########################################
    val_rewards, tr_rs = coo.evaluate()
    loss = -1 * np.mean(val_rewards[-5:]) * 1e6
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
    if loss < 0:
        f.writelines('$$$$$$$$$$$$$$$$$$$$$$' '\n')
    f.writelines('loss:'+str(loss)+'\n')
    f.writelines('eval_time:'+str(eval_time)+'\n')
    f.writelines('\n')