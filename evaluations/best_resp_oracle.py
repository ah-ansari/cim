import numpy as np


class Selection:
    def __init__(self, node):
        self.node = node
        self.cost_values = dict()


def create_selections(actions, mixed, values, max_cost):
    len_nodes = len(actions[0])
    len_actions = len(actions)

    selections = []
    for node_index in range(len_nodes):
        node_value = values[node_index]
        node_actions = actions[:, node_index]

        set_a = set(node_actions).union(set(node_actions+1))
        set_a.add(0)

        while max(set_a) > max_cost:
            set_a.remove(max(set_a))

        selection = Selection(node_index)

        for cost in set_a:
            v = 0
            for i in range(len_actions):
                if node_actions[i] > cost:
                    v -= mixed[i] * node_value
                elif cost > node_actions[i]:
                    v += mixed[i] * node_value

            selection.cost_values[cost] = v
        selections.append(selection)

    return selections


def dp(selections, K, len_nodes):
    table = np.zeros((len(selections)+1, K+1))
    table_c = np.zeros((len(selections)+1, K+1))

    for i in range(1, table.shape[0]):
        for k in range(0, table.shape[1]):
            s = selections[i - 1]
            max_ = -np.inf
            max_cost = 0
            for cost in s.cost_values.keys():
                if cost <= k:
                    value = s.cost_values[cost] + table[i - 1, k - cost]
                    if value >= max_:
                        max_ = value
                        max_cost = cost

            table[i, k] = max_
            table_c[i, k] = max_cost

    optimal_action = np.zeros(len_nodes, dtype=int)

    k = K
    for i in range(len(selections), 0, -1):
        optimal_action[selections[i-1].node] = table_c[i, k]
        k -= int(table_c[i, k])

    return optimal_action, table[len(selections), K]


def best_response(actions, mixed, values, k):
    consider_actions_index = np.where(mixed > 0)[0]
    consider_mixed = mixed[consider_actions_index]
    consider_actions = actions[consider_actions_index]

    seed_size = len(consider_actions[0])
    selections = create_selections(consider_actions, consider_mixed, values, k)
    optimal_action, v = dp(selections, k, seed_size)

    return optimal_action, v
