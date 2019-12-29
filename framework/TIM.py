import numpy as np
import networkx as nx
import math
import operator


def _random_rr_set(g: nx.Graph):
    free_nodes = g.graph['free']
    free_nodes_set = set(free_nodes)

    v_index = np.random.randint(len(g.graph["free"]))
    v = g.graph["free"][v_index]

    q = [v]
    rr_set = {v}

    while q:
        node = q.pop(0)
        for edge in g.in_edges(node, data=True):
            u = edge[0]
            if (u not in rr_set) and (u in free_nodes_set):
                r = np.random.random()
                if r < edge[2]['w']:
                    q.append(u)
                    rr_set.add(u)

    return rr_set


def create_rr_sets_graph(g: nx.Graph, theta):
    # creating theta number of random RR sets
    rr_sets = []
    for i in range(theta):
        rr_set = _random_rr_set(g)
        rr_sets.append(rr_set)
    # creating the bipartite cover graph, cover-graph shows which node covers which RR set
    rr_sets_graph = nx.DiGraph()
    for i in range(len(rr_sets)):
        for node in rr_sets[i]:
            rr_sets_graph.add_edge(node, "s" + str(i))

    return rr_sets_graph


def seed_selection(free_nodes, k, cover_graph):
    seed_set = []
    for i in range(k):
        degrees = cover_graph.degree(free_nodes)
        max_node = max(degrees.items(), key=operator.itemgetter(1))[0]
        seed_set.append(max_node)
        for edge in cover_graph.edges(max_node):
            cover_graph.remove_node(edge[1])

    return seed_set


def calculate_theta(g, k):
    kpt_star, r_p = _kpt_estimation(g, k)
    epsilon_p = 5.0 * np.power(1 * 0.01 / (k + 1), 1.0 / 3.0)
    kpt_plus = _refine_kpt(g, k, kpt_star, epsilon_p, r_p)

    Lambda = _get_lambda(g.number_of_nodes(), k)
    theta = Lambda / kpt_plus
    theta = int(theta)

    return theta


def _rr_set_weight(r, g:nx.Graph):
    result = 0
    for node in r:
        result += g.in_degree(node)

    return result


def _kpt_estimation(g: nx.Graph, k):
    n = g.number_of_nodes() * 1.0
    m = g.number_of_edges() * 1.0
    # l = 1.1282757460454809
    l = 1

    for i in range(1, int(np.log2(n))):
        rr_sets = []
        c = ((6.0*l*np.log(n)) + 6*np.log(np.log2(n))) * np.power(2.0, i*1.0)
        sum_ = 0
        for j in range(int(c)):
            r = _random_rr_set(g)
            # r = create_rr(g)
            rr_sets.append(r)
            k_r = 1.0 - np.power((1.0 - _rr_set_weight(r, g) / m), k)
            sum_ += k_r
        if sum_/c > 1.0/np.power(2.0, i):
            return n*sum_/(2.0*c), rr_sets
    return 1, []


def _get_lambda(n, k):
    # l = 1.1282757460454809
    l = 1
    eps = 0.1

    first = (8.0+2.0*eps)*n
    comb = math.factorial(n) / (math.factorial(k)*math.factorial(n-k))
    second = l*np.log(n)+np.log(comb)+np.log(2)
    result = first * second * np.power(eps, -2.0)
    return result


def _kpt_select_k_max(free_nodes, k, rr_sets):
    cover_graph = nx.DiGraph()

    for i in range(len(rr_sets)):
        for node in rr_sets[i]:
            cover_graph.add_edge(node, "s"+str(i))

    seed_set = []
    for i in range(k):
        degrees = cover_graph.degree(free_nodes)
        max_node = max(degrees.items(), key=operator.itemgetter(1))[0]
        seed_set.append(max_node)
        for edge in cover_graph.edges(max_node):
            cover_graph.remove_node(edge[1])

    return seed_set


def _refine_kpt(g: nx.Graph, k, kpt_star, epsilon_p, r_p):
    n = g.number_of_nodes()
    l = 1

    s_p = _kpt_select_k_max(g.graph["free"], k, r_p)

    lambda_p = (2+epsilon_p)*l*n*np.log(n)/(epsilon_p*epsilon_p)
    theta_p = lambda_p/kpt_star

    count = 0
    s_p = set(s_p)
    for i in range(int(theta_p)):
        r = _random_rr_set(g)
        if s_p.intersection(r):
            count += 1

    f = count/int(theta_p)
    kpt_p = f*n/(1+epsilon_p)

    return max(kpt_p, kpt_star)
