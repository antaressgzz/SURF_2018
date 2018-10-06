import numpy as np
from core.action_dis.action_discretization import action_discretization

def w2c(last_weight_array, actions):
    last_weight_array=last_weight_array.tolist()
    newls = []
    for k, x in actions.items():
        x=x.tolist()
        newls.append(x)
    actions=newls
    all_cost=[]
    for lw in last_weight_array:
        lw=np.array(lw)
        #print('lw=',lw)
        c=[]
        for action in actions:
            action=np.array(action)
            #print(action)
            cost = (np.abs(lw[1:] - action[1:])).sum()
            #print('cost:',cost)
            c.append(cost)
        all_cost.append(c)

    return all_cost

if __name__ == '__main__':
    lw=np.array([[0,1,0],[0,0,1]]);
    numb, actions=action_discretization(3,4)
    print(w2c(lw,actions))