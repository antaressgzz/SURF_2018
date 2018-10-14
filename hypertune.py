from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from coordinator import Coordinator
import tensorflow as tf
import numpy as np
import time
# %matplotlib inline

# Search space
param_space = {'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
               'activation': hp.choice('activation', [tf.nn.selu, tf.nn.relu, tf.nn.leaky_relu]),
               'fc_size': hp.choice('fc_size', [32, 64, 128, 256]),
               'window_length': hp.quniform('window_length', 50, 300, 50)}

# Define loss
def loss_processing(rewards, mode='inverse'):

    pass

# Number of model to train
max_evals = 2

# model name list
name_list = list(np.arange(max_evals))

# ob func
def val_reward(tuning_params):
    name = str(name_list.pop(0))
    print('Training model: '+name+'/'+str(max_evals-1))
    coo = Coordinator(tuning_params, name)
    val_rewards, tr_rs = coo.evaluate()
    # loss = loss_processing(val_rewards, mode='inverse')
    loss = -1 * np.mean(val_rewards[-4:]) * 1e6
    result = {'loss': loss,
              'status': STATUS_OK,
              'tr_rs': tr_rs,
              'val_rs': val_rewards,
              'params': tuning_params}

    return result

trails = Trials()
best = fmin(fn=val_reward,
            space=param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trails)

# Write the results in a txt file
f = open('./logs/'+time.strftime("%Y%m%d%H%M", time.localtime())+'.txt', 'w')
losses = []
for i in range(max_evals):
    dic = trails.results[i]
    f.writelines('model：'+str(i)+'\n')
    losses.append(dic['loss'])
    for key in dic.keys():
        f.writelines(key+': '+str(dic[key])+'\n')
f.writelines('Best model:'+str(np.argmin(losses))+' Params:'+str(best))
f.close()

