import numpy as np
import best_resp_oracle


def action_nash(nash, actions, nodes):
    action_index = np.random.choice(list(range(actions.shape[0])), p=nash)
    action = actions[action_index]

    action_dic = dict()
    for i in range(len(action)):
        action_dic[nodes[i]] = action[i]

    return action_dic


def action_v_each(k, v, nodes):
    action_dic = dict()
    for node_index in range(int(k/v)):
        action_dic[nodes[node_index]] = v

    if (k % v) != 0:
        action_dic[nodes[int(k/v)]] = k % v

    return action_dic


def action_random(k, nodes, max_value):    
    action_dic = dict()
    remained_budget = k
    node_index = 0

    while remained_budget > 1:
        v = np.random.randint(1, min(max_value+1, remained_budget))
        action_dic[nodes[node_index]] = v
        remained_budget -= v
        node_index += 1

    if remained_budget == 1:
        action_dic[nodes[node_index]] = 1

    return action_dic


def best_resp(action, k, nodes, values):
    if type(action) == dict:
        action = list(action.values())

    action_br = best_resp_oracle.best_response(np.array([action]), np.array([1]), values, k)[0]
    action_dic = dict()
    for i in range(len(action_br)):
        action_dic[nodes[i]] = action_br[i]

    return action_dic

