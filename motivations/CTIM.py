import numpy as np
import networkx as nx
import random
import operator


def _random_rr_set(g: nx.Graph, sb):
    free_nodes = g.graph['free']
    free_nodes_set = set(free_nodes)-set(sb)

    q = list(sb)
    B = set(sb)
    visited = set()

    while q:
        node = q.pop(0)
        B.add(node)
        for edge in g.out_edges(node, data=True):
            u = edge[0]
            if u not in visited:
                r = np.random.random()
                if r < edge[2]['w']:
                    q.append(u)
                    visited.add(u)

    v = random.sample(free_nodes_set, 1)[0]
    q = [v]
    visited = set()
    rr_set = set()

    while q:
        node = q.pop(0)
        rr_set.add(node)
        for edge in g.in_edges(node, data=True):
            u = edge[0]
            if (u not in rr_set) and (u not in B) and (u not in visited):
                r = np.random.random()
                if r < edge[2]['w']:
                    q.append(u)
                    visited.add(u)

    return rr_set


def create_rr_sets_graph(g: nx.Graph, theta, sb):
    # creating theta number of random RR sets
    rr_sets = []
    for i in range(theta):
        rr_set = _random_rr_set(g, sb)
        rr_sets.append(rr_set)
    # creating the bipartite cover graph, cover-graph shows which node covers which RR set
    rr_sets_graph = nx.DiGraph()
    for i in range(len(rr_sets)):
        for node in rr_sets[i]:
            if node == sb[0]:
                print(rr_sets[i])
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
